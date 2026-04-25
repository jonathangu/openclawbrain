import { describe, expect, it } from "vitest";
import { buildWeatherCapabilityChoiceRows, validateWeatherCapabilityChoiceLane, WEATHER_CAPABILITY_FAMILY, WEATHER_CAPABILITY_NODE_ID, WEATHER_INSTANCE_NODE_ID } from "../../scripts/validate-tool-capability-choice-lane.ts";

describe("tool capability choice weather lane", () => {
  it("selects the weather capability for fresh current-rain lookup and abstains for static definition", () => {
    const rows = buildWeatherCapabilityChoiceRows();
    const summary = validateWeatherCapabilityChoiceLane(rows);

    expect(summary).toMatchObject({
      family: WEATHER_CAPABILITY_FAMILY,
      pass: true,
      must_fire_pass: true,
      must_not_fire_pass: true,
    });

    const mustFire = rows.find((row) => row.episode_id === "ep_must_fire_current_weather");
    const mustNotFire = rows.find((row) => row.episode_id === "ep_must_not_fire_weather_definition");
    expect(mustFire).toBeDefined();
    expect(mustNotFire).toBeDefined();

    expect(mustFire).toMatchObject({
      chosen_action_kind: "tool_capability",
      chosen_tool_capability_id: WEATHER_CAPABILITY_FAMILY,
      chosen_tool_instance_id: null,
      stop_label: "CONTINUE",
    });
    expect(mustFire?.local_action_set.map((candidate) => candidate.action_kind)).toEqual([
      "traverse",
      "tool_capability",
      "tool_instance",
      "stop_local",
    ]);
    expect(mustFire?.hard_negatives).toContain(WEATHER_INSTANCE_NODE_ID);

    expect(mustNotFire).toMatchObject({
      chosen_action_kind: "stop_local",
      chosen_tool_capability_id: null,
      chosen_tool_instance_id: null,
      stop_label: "STOP_LOCAL",
    });
    expect(mustNotFire?.local_action_set.some((candidate) => candidate.node_id === WEATHER_CAPABILITY_NODE_ID && candidate.tool_capability_id === WEATHER_CAPABILITY_FAMILY)).toBe(true);
    expect(mustNotFire?.hard_negatives).toContain(WEATHER_CAPABILITY_NODE_ID);
    expect(mustNotFire?.hard_negatives).toContain(WEATHER_INSTANCE_NODE_ID);
  });
});
