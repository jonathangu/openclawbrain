declare function register(api: unknown): void;
declare const openclawbrainPlugin: {
    id: string;
    name: string;
    description: string;
    register: typeof register;
};
export default openclawbrainPlugin;
