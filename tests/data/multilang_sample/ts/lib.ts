export interface Person {
  name: string;
}

export function greet(person: Person): string {
  return `hello ${person.name}`;
}
