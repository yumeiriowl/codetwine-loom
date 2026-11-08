#!/usr/bin/env node
// The default static check for JavaScript. Runs ESLint's no-undef rule, with the
// settings in CTLOOM_CHECK_HOME and not the repository's own, over the .js, .jsx, .mjs
// and .cjs files, whatever the case of the extension, that git tracks or does not
// ignore in the current directory and that are files or links to files that can be
// read, and prints one "<path>:<line>:<column>: <message> [<rule>]" line per finding. A
// file ESLint cannot parse is one finding. What ESLint reports besides no-undef and
// parsing errors (rules a comment in a file turns on, rules a comment names that ESLint
// does not have, /* eslint-env */ comments) is not printed.
//
// Exits 127 when ESLint is not installed under CTLOOM_CHECK_HOME or fails to run.
"use strict";

const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const HOME = process.env.CTLOOM_CHECK_HOME || "/opt/ctloom-check";
const EXT_SET = new Set([".js", ".jsx", ".mjs", ".cjs"]);
const RULE = "no-undef";
const PARSE_ERROR_HEAD = "Parsing error:";
const CANNOT_RUN = 127;

/** Say whether a path is a file, or a link to one, that can be read. */
function isReadableFile(file) {
  try {
    const stat = fs.statSync(file, { throwIfNoEntry: false });
    if (!stat || !stat.isFile()) {
      return false;
    }
    fs.accessSync(file, fs.constants.R_OK);
    return true;
  } catch {
    return false;
  }
}

/** Return the files git tracks or does not ignore, that are files or links to files
 * that can be read and have one of extSet. */
function gitFileList(extSet) {
  const out = execFileSync("git", ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                           { encoding: "utf8", maxBuffer: 1 << 30 });
  const fileSet = new Set(out.split("\0").filter(Boolean));
  return [...fileSet].filter(file => extSet.has(path.extname(file).toLowerCase())
                                     && isReadableFile(file));
}

async function main() {
  let ESLint;
  try {
    ({ ESLint } = require(path.join(HOME, "node_modules", "eslint")));
  } catch (error) {
    console.error(`ctloom-check-js: ESLint is not installed under ${HOME}: ${error.message}`);
    return CANNOT_RUN;
  }
  const fileList = gitFileList(EXT_SET);
  if (!fileList.length) {
    return 0;
  }
  const eslint = new ESLint({
    cwd: process.cwd(),
    overrideConfigFile: path.join(HOME, "eslint.config.mjs"),
    errorOnUnmatchedPattern: false,
    warnIgnored: false,
  });
  const resultList = await eslint.lintFiles(fileList);
  for (const result of resultList) {
    const file = path.relative(process.cwd(), result.filePath);
    for (const message of result.messages) {
      const isParseError = message.fatal
                           && message.message.startsWith(PARSE_ERROR_HEAD);
      if (message.ruleId !== RULE && !isParseError) {
        continue;
      }
      console.log(`${file}:${message.line || 1}:${message.column || 1}: `
                  + `${message.message} [${isParseError ? "parse" : RULE}]`);
    }
  }
  return 0;
}

main().then(
  code => process.exit(code),
  error => {
    console.error(`ctloom-check-js: ESLint failed: ${error.stack || error}`);
    process.exit(CANNOT_RUN);
  });
