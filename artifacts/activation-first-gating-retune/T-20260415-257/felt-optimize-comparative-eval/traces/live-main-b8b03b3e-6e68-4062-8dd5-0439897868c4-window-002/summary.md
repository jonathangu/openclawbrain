# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68b2fa5dad518df8f3866150e5eb5ae2df4d40d0d1730a7b39326babb425756a`
- fixture hash: `sha256-9c558cf390c2d5519271f6ba91a97c5aab0727de8cfbeaa1362c2e39d2a00c20`
- score hash: `sha256-c74af4cece32a619aec5c228034c45a8cdd8ba24e0392a5893e19678e38b8d72`
- bundle hash: `sha256-6cb3467bdf1870de20bf803413c9cdcb5b066a4ba0e40b0a7d19581328bd552a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8709c6589a780862225e4afa90cbbf44ed4ef4f7b39772bdc54c0a9f8a33087 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-617f99652c0ad99472fa02ce34072064e07c43f5153bfe7a0dc635b361a41b20 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4ede7a1dd4eb037591894d6d56d735197992985a6fd88175705de64be5b2ce1f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-99b6830cffd9174d769292ad053e215cb0c0c8373dc23b64bd49f6a9c22d127c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ef22ba33 | sha256-22e58771f3ebc4c1fdceb21737f50672cf5f7d9af04a0621805c962f05d14ce3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ef22ba33 | sha256-eac9481c044f264f303cae5db9238ad9b857c536c18d1726123d1182e94161e7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5be09020 | sha256-2629fe3e00f377c592ba8e1e9a3f9524d4341d961a1794df450c09133b8b933b |
