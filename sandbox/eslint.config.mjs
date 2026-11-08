// The settings ctloom-check-js runs ESLint with: no-undef alone, over JavaScript and
// JSX whatever the case of the extension, with the globals of the browser, Node.js, Bun
// and the jest and mocha test runners defined. A .cjs file is read as CommonJS.
import globals from "globals";

const isJavaScript = path => /\.(?:[cm]?js|jsx)$/i.test(path);
const isCommonJs = path => /\.cjs$/i.test(path);

export default [
  {
    files: [isJavaScript],
    languageOptions: {
      ecmaVersion: "latest",
      parserOptions: { ecmaFeatures: { jsx: true } },
      globals: {
        ...globals.browser, ...globals.node, ...globals.jest, ...globals.mocha, Bun: "readonly",
      },
    },
    linterOptions: { reportUnusedDisableDirectives: "off" },
    rules: { "no-undef": "error" },
  },
  {
    files: [isCommonJs],
    languageOptions: { sourceType: "commonjs" },
  },
];
