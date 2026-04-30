declare const process: { pid: number; platform: string; execPath: string; argv: string[]; stderr: { write(value: string): void }; stdout: { write(value: string): void }; exit(code?: number): never };

declare module 'node:os' {
  export function homedir(): string;
  export function tmpdir(): string;
}

declare module 'node:path' {
  const path: {
    join(...parts: string[]): string;
    resolve(...parts: string[]): string;
    dirname(value: string): string;
  };
  export default path;
}

declare module 'node:fs/promises' {
  export function chmod(path: string, mode: number): Promise<void>;
  export function mkdir(path: string, options?: any): Promise<void>;
  export function lstat(path: string): Promise<any>;
  export function readFile(path: string, encoding?: string): Promise<string>;
  export function rename(oldPath: string, newPath: string): Promise<void>;
  export function rm(path: string, options?: any): Promise<void>;
  export function symlink(target: string, path: string): Promise<void>;
  export function writeFile(path: string, data: string, options?: any): Promise<void>;
}

declare module 'node:crypto' {
  export function randomUUID(): string;
  export function createHash(algorithm: string): { update(value: string): { digest(encoding: 'hex'): string } };
}
