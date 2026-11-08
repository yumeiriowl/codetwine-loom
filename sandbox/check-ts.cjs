#!/usr/bin/env node
// The default static check for TypeScript. Type-checks the repository in the current
// directory with the TypeScript under CTLOOM_CHECK_HOME, emitting nothing and writing
// nothing, and prints one "<path>(<line>,<column>): error TS<n>: <message>" line per
// finding in a file. Errors that belong to no file go to stderr and are not findings.
//
// With a tsconfig.json at the top that can be read, that configuration and every
// configuration it references that can be read are checked, each once. Without one, the
// .ts, .tsx, .mts and .cts files git tracks or does not ignore, and that are files or
// links to files that can be read, are checked with settings that accept imports
// without an extension and JSX. Either way the type definitions of Node.js and jest
// under CTLOOM_CHECK_HOME, and Bun's in a repository run by Bun (a bun.lock, a bun.lockb
// or a bunfig.toml, a packageManager of bun or a script of package.json that runs bun or
// bunx), are read along with the configuration's own: those its types names, or without
// types every package under its typeRoots (node_modules/@types by default).
//
// A package that cannot be found or has no type definitions is not a finding; a relative
// or absolute module that cannot be found is.
//
// Exits 127 when TypeScript is not installed under CTLOOM_CHECK_HOME or fails to run.
"use strict";

const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const HOME = process.env.CTLOOM_CHECK_HOME || "/opt/ctloom-check";
const CANNOT_RUN = 127;
const ADDED_TYPE_LIST = ["node", "jest"];
const BUN_TYPE = "bun";
const BUN_FILE_LIST = ["bun.lock", "bun.lockb", "bunfig.toml"];
// bun or bunx as a command in a script of package.json
const BUN_COMMAND_RE = /(?:^|[\s;&|(])bunx?(?=\s|$)/;
// TS2307, TS2792: a module that cannot be found. TS7016: a module with no type definitions
const MODULE_NOT_FOUND_SET = new Set([2307, 2792]);
const NO_TYPE_DEFINITION = 7016;
const FALLBACK_OPTION_DICT = {
  target: "esnext", module: "preserve", moduleResolution: "bundler", jsx: "preserve",
  allowImportingTsExtensions: true, esModuleInterop: true, skipLibCheck: true,
};

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

/** Say whether the repository in the current directory is run by Bun. */
function isBunProject() {
  if (BUN_FILE_LIST.some(isReadableFile)) {
    return true;
  }
  let pkg;
  try {
    pkg = JSON.parse(fs.readFileSync("package.json", "utf8"));
  } catch {
    return false;
  }
  if (!pkg || typeof pkg !== "object") {
    return false;
  }
  if (typeof pkg.packageManager === "string" && pkg.packageManager.startsWith("bun@")) {
    return true;
  }
  const scriptDict = pkg.scripts && typeof pkg.scripts === "object" ? pkg.scripts : {};
  return Object.values(scriptDict).some(
    script => typeof script === "string" && BUN_COMMAND_RE.test(script));
}

/** Return the type definitions this check adds: ADDED_TYPE_LIST, and BUN_TYPE in a
 * repository run by Bun. */
function addedTypeList() {
  return isBunProject() ? [...ADDED_TYPE_LIST, BUN_TYPE] : ADDED_TYPE_LIST;
}

/** Return the .ts, .tsx, .mts and .cts files git tracks or does not ignore, that are
 * files or links to files that can be read. */
function gitFileList() {
  const out = execFileSync("git", ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                           { encoding: "utf8", maxBuffer: 1 << 30 });
  return [...new Set(out.split("\0").filter(Boolean))]
    .filter(file => /\.(tsx?|mts|cts)$/i.test(file) && isReadableFile(file))
    .map(file => path.resolve(file));
}

/** Return options with nothing to write and this check's type definitions (typeList)
 * added. */
function checkOptionDict(ts, optionDict, typeList) {
  const own = { ...optionDict };
  for (const key of ["incremental", "composite", "tsBuildInfoFile"]) {
    delete own[key];
  }
  const typeRootList = own.typeRoots || [path.resolve("node_modules", "@types")];
  const ownTypeList = own.types
    || ts.getAutomaticTypeDirectiveNames({ typeRoots: typeRootList }, ts.sys);
  return {
    ...own,
    noEmit: true,
    types: [...new Set([...ownTypeList, ...typeList])],
    typeRoots: [...new Set([...typeRootList, path.join(HOME, "node_modules", "@types")])],
  };
}

/** Return whether a diagnostic is about a package the container does not have. */
function isMissingPackage(ts, diagnostic) {
  if (diagnostic.code === NO_TYPE_DEFINITION) {
    return true;
  }
  if (!MODULE_NOT_FOUND_SET.has(diagnostic.code)) {
    return false;
  }
  const head = ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n").split("\n")[0];
  const match = head.match(/'([^']+)'/);
  return Boolean(match) && !match[1].startsWith(".") && !match[1].startsWith("/");
}

/** Write one diagnostic: a finding line when it has a file, stderr otherwise. */
function report(ts, diagnostic, lineSet) {
  if (diagnostic.category !== ts.DiagnosticCategory.Error || isMissingPackage(ts, diagnostic)) {
    return;
  }
  const head = ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n").split("\n")[0];
  if (!diagnostic.file || diagnostic.start === undefined) {
    console.error(`ctloom-check-ts: error TS${diagnostic.code}: ${head}`);
    return;
  }
  const { line, character } = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
  const file = path.relative(process.cwd(), diagnostic.file.fileName);
  lineSet.add(`${file}(${line + 1},${character + 1}): error TS${diagnostic.code}: ${head}`);
}

/** Return what a file holds, or undefined when it is not a file that can be read. */
function readReadableFile(ts, file) {
  return isReadableFile(file) ? ts.sys.readFile(file) : undefined;
}

/** Type-check rootNames with optionDict and the type definitions of typeList, and add
 * what it finds to lineSet. Only files that can be read are read. */
function checkProgram(ts, rootNames, optionDict, typeList, lineSet) {
  if (!rootNames.length) {
    return;
  }
  const options = checkOptionDict(ts, optionDict, typeList);
  const host = ts.createCompilerHost(options);
  host.readFile = file => readReadableFile(ts, file);
  host.fileExists = isReadableFile;
  const program = ts.createProgram({ rootNames, options, host });
  for (const diagnostic of [...program.getOptionsDiagnostics(), ...program.getGlobalDiagnostics(),
                            ...program.getSyntacticDiagnostics(),
                            ...program.getSemanticDiagnostics()]) {
    report(ts, diagnostic, lineSet);
  }
}

/** Check a tsconfig.json and every configuration it references, each once, with the
 * type definitions of typeList. Only configurations that can be read are read. */
function checkConfig(ts, configPath, typeList, lineSet) {
  const seenSet = new Set();
  const queue = [path.resolve(configPath)];
  while (queue.length) {
    const current = queue.shift();
    if (seenSet.has(current) || !isReadableFile(current)) {
      continue;
    }
    seenSet.add(current);
    const parsed = ts.getParsedCommandLineOfConfigFile(current, {}, {
      ...ts.sys,
      readFile: file => readReadableFile(ts, file),
      onUnRecoverableConfigFileDiagnostic: diagnostic => report(ts, diagnostic, lineSet),
    });
    if (!parsed) {
      continue;
    }
    parsed.errors.forEach(diagnostic => report(ts, diagnostic, lineSet));
    for (const reference of parsed.projectReferences || []) {
      queue.push(ts.resolveProjectReferencePath(reference));
    }
    checkProgram(ts, parsed.fileNames, parsed.options, typeList, lineSet);
  }
}

function main() {
  let ts;
  try {
    ts = require(path.join(HOME, "node_modules", "typescript"));
  } catch (error) {
    console.error(`ctloom-check-ts: TypeScript is not installed under ${HOME}: ${error.message}`);
    return CANNOT_RUN;
  }
  const lineSet = new Set();
  const typeList = addedTypeList();
  if (isReadableFile("tsconfig.json")) {
    checkConfig(ts, "tsconfig.json", typeList, lineSet);
  } else {
    const converted = ts.convertCompilerOptionsFromJson(FALLBACK_OPTION_DICT, process.cwd());
    checkProgram(ts, gitFileList(), converted.options, typeList, lineSet);
  }
  for (const line of lineSet) {
    console.log(line);
  }
  return 0;
}

try {
  process.exit(main());
} catch (error) {
  console.error(`ctloom-check-ts: TypeScript failed: ${error.stack || error}`);
  process.exit(CANNOT_RUN);
}
