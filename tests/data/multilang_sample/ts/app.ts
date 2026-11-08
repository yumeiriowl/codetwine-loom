import { greet, Person } from './lib';

export function run(name: string): string {
  const person: Person = { name };
  return greet(person);
}
