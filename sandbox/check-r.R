#!/usr/bin/env Rscript
# The default static check for R. Reads every R script of the repository in the current
# directory that git tracks or does not ignore (.R, .r) and that can be read, is not a
# directory and is not empty, and prints one "<path>:<line>[:<column>]: <message>"
# line for
#   - each script R cannot parse (the syntax error), and
#   - each call of a function no script of the repository defines and no package the
#     repository attaches provides ("no visible global function definition for 'name'").
#
# No code of the repository is run: a script is parsed, and each of its top-level
# expressions is checked as the body of a function that is never called
# (codetools::checkUsage). The names every script defines at its top level (also inside
# an if or a { } written there), the names setGeneric / setClass / assign and the like
# are given as strings, the names given to globalVariables() as strings, and the names a
# NAMESPACE file imports with importFrom are taken as defined. The packages named by
# library() / require() calls, by Depends of a DESCRIPTION file and by import() of a
# NAMESPACE file are attached, and testthat for the scripts under a testthat directory.
# When one of those packages is not installed, the calls of undefined functions are not
# reported: of that script alone when the script names the package, of every script when
# DESCRIPTION or NAMESPACE names it. Syntax errors are always reported. A DESCRIPTION or
# NAMESPACE file that cannot be read, is a directory or is empty is not read, and one
# that cannot be parsed is ignored.
#
# Exits 127 when the package codetools is not installed.

SCRIPT_PATTERN <- "\\.[Rr]$"
TEST_DIR_NAME <- "testthat"
ATTACH_FUNCTION_VECTOR <- c("library", "require", "requireNamespace", "loadNamespace")
NAME_STRING_FUNCTION_VECTOR <- c(
  "assign", "delayedAssign", "makeActiveBinding", "setGeneric", "setClass",
  "setRefClass", "setClassUnion", "setMethod", "setReplaceMethod", "setValidity"
)
ASSIGN_OPERATOR_VECTOR <- c("<-", "=", "<<-")
BLOCK_FUNCTION_VECTOR <- c("if", "{")
GLOBAL_NAME_FUNCTION <- "globalVariables"
UNDEFINED_FUNCTION_TEXT <- "no visible global function definition for"

# Say, for each path, whether it can be read, is not a directory and holds at least one
# byte.
is_readable_file <- function(path_vector) {
  info <- file.info(path_vector, extra_cols = FALSE)
  !is.na(info$isdir) & !info$isdir & info$size > 0 & file.access(path_vector, 4) == 0
}

# Return the R scripts git tracks or does not ignore that can be read, are not
# directories and hold at least one byte.
script_path_vector <- function() {
  path_vector <- suppressWarnings(
    system2(
      "git", c("-c", "core.quotepath=off", "ls-files", "-co", "--exclude-standard"),
      stdout = TRUE, stderr = FALSE
    )
  )
  if (!is.null(attr(path_vector, "status"))) {
    path_vector <- list.files(".", recursive = TRUE)
  }
  path_vector <- path_vector[grepl(SCRIPT_PATTERN, path_vector)]
  sort(path_vector[is_readable_file(path_vector)])
}

# Parse a script. Return its expressions, or print its syntax error as one finding and
# return NULL.
parse_script <- function(path) {
  tryCatch(
    parse(path, keep.source = TRUE),
    error = function(error) {
      text <- strsplit(conditionMessage(error), "\n")[[1]][1]
      match <- regmatches(text, regexec(":([0-9]+):([0-9]+): (.*)$", text))[[1]]
      if (length(match) == 4) {
        cat(sprintf("%s:%s:%s: %s\n", path, match[2], match[3], match[4]))
      } else {
        cat(sprintf("%s:1: %s\n", path, text))
      }
      NULL
    }
  )
}

# Return the name of the function a call calls: "f" for f(...) and for pkg::f(...), ""
# for any other call.
call_name <- function(expression) {
  head <- expression[[1]]
  if (is.name(head)) {
    return(as.character(head))
  }
  if (is.call(head) && length(head) == 3 && is.name(head[[1]]) &&
        as.character(head[[1]]) %in% c("::", ":::") && is.name(head[[3]])) {
    return(as.character(head[[3]]))
  }
  ""
}

# Return the arguments of a call, less the empty ones (the first index of x[, 1]).
call_part_list <- function(expression) {
  part_list <- as.list(expression)[-1]
  part_list[vapply(seq_along(part_list),
                   function(index) !identical(part_list[[index]], quote(expr = )),
                   logical(1))]
}

# Return the names a top-level expression defines: the target of an assignment, the
# name given as a string to a NAME_STRING_FUNCTION_VECTOR function, and the names the
# expressions inside an if or a { } define.
defined_name_vector <- function(expression) {
  if (!is.call(expression)) {
    return(character(0))
  }
  function_name <- call_name(expression)
  if (function_name %in% ASSIGN_OPERATOR_VECTOR && length(expression) >= 3) {
    target <- expression[[2]]
    if (is.name(target) || is.character(target)) {
      return(as.character(target))
    }
  }
  if (function_name %in% NAME_STRING_FUNCTION_VECTOR && length(expression) >= 2 &&
        is.character(expression[[2]])) {
    return(expression[[2]])
  }
  if (function_name %in% BLOCK_FUNCTION_VECTOR) {
    return(unlist(lapply(call_part_list(expression), defined_name_vector)))
  }
  character(0)
}

# Return the names the globalVariables() calls of an expression are given as strings,
# at any depth.
global_name_vector <- function(expression) {
  if (!is.call(expression)) {
    return(character(0))
  }
  name_vector <- character(0)
  if (call_name(expression) == GLOBAL_NAME_FUNCTION && length(expression) >= 2) {
    argument <- expression[[2]]
    part_list <- if (is.call(argument)) call_part_list(argument) else list(argument)
    name_vector <- unlist(Filter(is.character, part_list))
  }
  for (part in call_part_list(expression)) {
    if (is.call(part)) {
      name_vector <- c(name_vector, global_name_vector(part))
    }
  }
  name_vector
}

# Return the package names the attach calls of an expression name, at any depth.
attach_package_vector <- function(expression) {
  if (!is.call(expression)) {
    return(character(0))
  }
  name_vector <- character(0)
  function_name <- call_name(expression)
  if (function_name %in% ATTACH_FUNCTION_VECTOR && length(expression) >= 2) {
    argument <- expression[[2]]
    if (is.name(argument) || is.character(argument)) {
      name_vector <- as.character(argument)
    }
  }
  for (part in call_part_list(expression)) {
    if (is.call(part)) {
      name_vector <- c(name_vector, attach_package_vector(part))
    }
  }
  name_vector
}

# Return the packages the DESCRIPTION and NAMESPACE files at the top attach, and the
# names NAMESPACE imports one by one, as list(package = , name = ). A file that cannot
# be read, is a directory or is empty is not read, and one that cannot be parsed is
# ignored.
package_setting <- function() {
  package_vector <- character(0)
  name_vector <- character(0)
  if (is_readable_file("DESCRIPTION")) {
    depend <- tryCatch(
      read.dcf("DESCRIPTION", fields = "Depends"),
      error = function(error) matrix(character(0), 0, 1)
    )
    text <- paste(depend[!is.na(depend)], collapse = ",")
    package_vector <- trimws(gsub("\\(.*?\\)", "", strsplit(text, ",")[[1]]))
    package_vector <- setdiff(package_vector[nzchar(package_vector)], "R")
  }
  if (is_readable_file("NAMESPACE")) {
    directive_list <- tryCatch(as.list(parse("NAMESPACE")), error = function(error) list())
    for (directive in directive_list) {
      if (!is.call(directive)) {
        next
      }
      argument_vector <- vapply(
        as.list(directive)[-1], function(argument) paste(deparse(argument), collapse = ""),
        character(1)
      )
      argument_vector <- gsub("^[\"']|[\"']$", "", argument_vector)
      directive_name <- as.character(directive[[1]])
      if (directive_name == "import") {
        package_vector <- c(package_vector, argument_vector)
      } else if (directive_name == "importFrom" && length(argument_vector) > 1) {
        name_vector <- c(name_vector, argument_vector[-1])
      }
    }
  }
  list(package = unique(package_vector), name = unique(name_vector))
}

# Attach the packages given. Return the names of the ones that are not installed.
attach_package <- function(package_vector) {
  missing_vector <- character(0)
  for (package in unique(package_vector)) {
    is_attached <- suppressWarnings(suppressPackageStartupMessages(
      require(package, character.only = TRUE, quietly = TRUE)
    ))
    if (!is_attached) {
      missing_vector <- c(missing_vector, package)
    }
  }
  missing_vector
}

# Print the calls of undefined functions in the expressions of one script. A finding is
# given the line its message names, else the first line of its top-level expression.
report_undefined_function <- function(path, script, environment) {
  finding_vector <- character(0)
  collect <- function(text) {
    finding_vector <<- c(finding_vector, text)
  }
  source_reference_list <- attr(script, "srcref")
  for (index in seq_along(script)) {
    expression <- script[[index]]
    checked <- tryCatch(
      eval(call("function", NULL, expression), envir = environment),
      error = function(error) NULL
    )
    if (is.null(checked)) {
      next
    }
    line <- if (index <= length(source_reference_list)) {
      as.integer(source_reference_list[[index]][1])
    } else {
      1L
    }
    try(
      codetools::checkUsage(checked, name = "", report = function(text) {
        collect(sprintf("%s\t%d", text, line))
      }, all = FALSE, suppressLocal = TRUE),
      silent = TRUE
    )
  }
  for (finding in unique(finding_vector)) {
    part_vector <- strsplit(finding, "\t")[[1]]
    text <- gsub("[‘’]", "'", trimws(part_vector[1]))
    if (!grepl(UNDEFINED_FUNCTION_TEXT, text, fixed = TRUE)) {
      next
    }
    line_match <- regmatches(text, regexec(":([0-9]+)(-[0-9]+)?\\)$", text))[[1]]
    line <- if (length(line_match) >= 2) line_match[2] else part_vector[2]
    name_match <- regmatches(text, regexec("for '([^']+)'", text))[[1]]
    cat(sprintf("%s:%s: %s '%s'\n", path, line, UNDEFINED_FUNCTION_TEXT, name_match[2]))
  }
}

main <- function() {
  if (!requireNamespace("codetools", quietly = TRUE)) {
    message("ctloom-check-r: the package codetools is not installed")
    quit(save = "no", status = 127)
  }
  path_vector <- script_path_vector()
  script_list <- list()
  for (path in path_vector) {
    expression_list <- parse_script(path)
    if (!is.null(expression_list)) {
      script_list[[path]] <- expression_list
    }
  }

  setting <- package_setting()
  defined_vector <- setting$name
  for (expression_list in script_list) {
    defined_vector <- c(
      defined_vector,
      unlist(lapply(as.list(expression_list), defined_name_vector)),
      unlist(lapply(as.list(expression_list), global_name_vector))
    )
  }
  if (length(attach_package(setting$package)) > 0) {
    return(invisible(NULL))
  }

  for (path in names(script_list)) {
    expression_list <- as.list(script_list[[path]])
    package_vector <- unlist(lapply(expression_list, attach_package_vector))
    if (TEST_DIR_NAME %in% strsplit(path, "/")[[1]]) {
      package_vector <- c(package_vector, "testthat")
    }
    if (length(attach_package(package_vector)) > 0) {
      next
    }
    environment <- new.env(parent = globalenv())
    for (name in unique(defined_vector)) {
      assign(name, function(...) NULL, envir = environment)
    }
    report_undefined_function(path, script_list[[path]], environment)
  }
}

main()
