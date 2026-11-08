#!/usr/bin/env Rscript
# The default test command for R. Runs the testthat tests of the repository in the
# current directory with the "check" reporter.
#
# Usage: ctloom-test-r [test file or directory]...
#
# With a DESCRIPTION file at the top the repository is a package: the packages it names
# in Depends, Imports, LinkingTo and Suggests that are not installed are installed into
# the user library (R_LIBS_USER, under HOME) from the repositories of the "repos" option
# (https://cloud.r-project.org when it names none), unless CTLOOM_TEST_R_OFFLINE is set, and
# the tests run with the package loaded from its sources (testthat::test_local). Without one the files under
# tests/testthat run as they are (testthat::test_dir). With arguments only those run: a
# test file with testthat::test_file, the test files of a directory with
# testthat::test_dir.
#
# Exits 0 when every test passes and 1 when a test fails or errors. Exits 1 as well, with
# the reason on stderr and no test run, when testthat is not installed, when there is no
# tests/testthat directory, when a path given does not exist, or when a package the
# DESCRIPTION file names in Depends, Imports or LinkingTo is not installed and cannot be
# installed.

TEST_DIR <- "tests/testthat"
DESCRIPTION_FILE <- "DESCRIPTION"
NEED_FIELD_VECTOR <- c("Depends", "Imports", "LinkingTo")
WANT_FIELD_VECTOR <- c("Suggests")
DEFAULT_REPOS <- "https://cloud.r-project.org"
OFFLINE_VARIABLE <- "CTLOOM_TEST_R_OFFLINE"

# Stop the script with a reason on stderr and the status that says no test ran.
quit_without_run <- function(...) {
  message("ctloom-test-r: ", ...)
  quit(save = "no", status = 1)
}

# Return the package names the given fields of the DESCRIPTION file name, without "R"
# and without their version requirements.
description_package_vector <- function(field_vector) {
  description <- read.dcf(DESCRIPTION_FILE, fields = field_vector)
  text <- paste(description[!is.na(description)], collapse = ",")
  name_vector <- trimws(gsub("\\(.*?\\)", "", strsplit(text, ",")[[1]]))
  setdiff(name_vector[nzchar(name_vector)], "R")
}

# Return the names among the given packages that are not installed.
missing_package_vector <- function(name_vector) {
  name_vector[!vapply(name_vector, requireNamespace, logical(1), quietly = TRUE)]
}

# Install the packages the DESCRIPTION file names that are not installed into the user
# library, and stop without a run when one the package needs is still missing.
install_dependency <- function() {
  user_library <- Sys.getenv("R_LIBS_USER")
  if (nzchar(user_library)) {
    dir.create(user_library, recursive = TRUE, showWarnings = FALSE)
    .libPaths(c(user_library, .libPaths()))
  }
  need_vector <- description_package_vector(NEED_FIELD_VECTOR)
  missing_vector <- missing_package_vector(
    c(need_vector, description_package_vector(WANT_FIELD_VECTOR))
  )
  if (length(missing_vector) > 0 && !nzchar(Sys.getenv(OFFLINE_VARIABLE))) {
    repos <- getOption("repos")
    if (is.null(repos) || any(repos == "@CRAN@")) {
      repos <- DEFAULT_REPOS
    }
    try(
      suppressWarnings(utils::install.packages(
        missing_vector, lib = .libPaths()[1], repos = repos, quiet = TRUE
      )),
      silent = TRUE
    )
  }
  missing_vector <- missing_package_vector(need_vector)
  if (length(missing_vector) > 0) {
    quit_without_run(
      "these packages are not installed and could not be installed: ",
      paste(missing_vector, collapse = ", ")
    )
  }
}

main <- function() {
  path_vector <- commandArgs(trailingOnly = TRUE)
  if (!requireNamespace("testthat", quietly = TRUE)) {
    quit_without_run("the package testthat is not installed")
  }
  is_package <- file.exists(DESCRIPTION_FILE)
  missing_path_vector <- path_vector[!file.exists(path_vector)]
  if (length(missing_path_vector) > 0) {
    quit_without_run("no such test file: ", paste(missing_path_vector, collapse = ", "))
  }
  if (length(path_vector) == 0 && !dir.exists(TEST_DIR)) {
    quit_without_run(
      "there is no ", TEST_DIR, " in ", getwd(),
      "; set TEST_COMMAND to run the tests of a repository laid out another way"
    )
  }
  if (is_package) {
    install_dependency()
  }

  if (length(path_vector) > 0) {
    load_package <- if (is_package) "source" else "none"
    for (path in path_vector) {
      run_test <- if (dir.exists(path)) testthat::test_dir else testthat::test_file
      run_test(path, reporter = "check", stop_on_failure = TRUE, load_package = load_package)
    }
  } else if (is_package) {
    testthat::test_local(reporter = "check", stop_on_failure = TRUE)
  } else {
    testthat::test_dir(TEST_DIR, reporter = "check", stop_on_failure = TRUE)
  }
}

main()
