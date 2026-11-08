import { greet } from './lib';

export function run(name) {
  return greet(name).toUpperCase();
}
