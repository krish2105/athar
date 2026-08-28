/** Every number on this page is read from a committed metrics artifact.
 *
 * `import.meta.glob` rather than named imports, because the artifacts appear as
 * their build steps run and a missing one must degrade the page rather than break
 * the build. `available()` is what each section checks before rendering.
 *
 * No figure in this application is typed by hand. If a number is not in one of
 * these files, it does not appear.
 */

type Json = Record<string, any>;

const loaded = import.meta.glob("../../metrics/*.json", { eager: true }) as Record<
  string,
  { default: Json }
>;

const artifacts: Record<string, Json> = {};
for (const [path, module] of Object.entries(loaded)) {
  const name = path.split("/").pop()!.replace(".json", "");
  artifacts[name] = module.default;
}

export function artifact(name: string): Json | null {
  return artifacts[name] ?? null;
}

export function available(...names: string[]): boolean {
  return names.every((name) => artifacts[name] != null);
}

export type Provenance = {
  source: string;
  synthetic: boolean;
  split?: string;
  seed?: number;
  dgp_hash?: string;
  caveat?: string;
};

/** The caveat is derived from the artifact, never written into a component.
 *  A chart backed by simulated data cannot be shipped without one. */
export function provenance(name: string): Provenance | null {
  const block = artifacts[name]?.provenance;
  return block ? (block as Provenance) : null;
}

export const NAMES: Record<string, string> = {
  search_brand: "Brand search",
  search_nonbrand: "Non-brand search",
  social_paid: "Paid social",
  display_prog: "Programmatic display",
  video_ctv: "Video / CTV",
};

export function channelLabel(key: string): string {
  return NAMES[key] ?? key;
}
