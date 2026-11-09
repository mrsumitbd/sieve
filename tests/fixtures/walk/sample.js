import fs from "fs";
const path = require("path");

/** Plain declaration. */
function plain(a, { b, c }, ...rest) {
  function nested() { return 1; }
  const arrowInside = () => 2;
  return nested() + arrowInside();
}

async function asyncFn() { return await 1; }
function* gen() { yield 1; }
export function exported(x) { return x; }
export default function () { return 0; }
export const exportedArrow = (y) => y * 2;

const arrow = (x) => x + 1;
const single = x => x;
const block = (a, b) => { return a + b; };
let late = () => {}, also = async () => {};
var oldStyle = function () { return 1; };
const named = function inner() { return 2; };

[1, 2, 3].map(x => x * 2);
setTimeout(function () { console.log("later"); }, 10);
const withCallback = items.filter((i) => i.ok);
const obj = {
  method() { return 1; },
  prop: () => 2,
  get getter() { return 3; },
};

(function iife() { return 4; })();
(() => 5)();

/** Class doc. */
class Animal extends Base {
  constructor(name) { super(); this.name = name; }
  speak() { return `${this.name}`; }
  static create() { return new Animal("x"); }
  get label() { return this.name; }
  field = () => 1;
  async load() { return [1].map(v => v); }
}

const Expr = class { m() {} };
class Outer { m() { class Inner { n() {} } return Inner; } }

if (fs) {
  function conditional() { return 6; }
}
module.exports = { plain, Animal };
