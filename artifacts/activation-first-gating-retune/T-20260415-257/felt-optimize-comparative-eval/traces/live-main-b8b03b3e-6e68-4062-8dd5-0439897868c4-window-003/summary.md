# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a2590462dc987ced28ec91e593a00f4b408387f6ec40a92d626a6087fcbd75f`
- fixture hash: `sha256-aace8a3fe4087409ebd528569ab1ac34f47ecd7317117709f7ec2907eaa6127c`
- score hash: `sha256-a5b0b42ef933470a35a1ff7cac37da98eecc30824aaddf6d4016a7ddc5674836`
- bundle hash: `sha256-47d6fbfe62454c363382117d4883393617d7d60112edfb6ad76f5112872c633b`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b2d2d9dd5ce486e4334796b2692780e0b5a1aabacd13eeb32d1dca3c57b5e799 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-566edd9bc0ff39d49f43f9c12d31e3c6a0ad01fb8fe5515e7f446867463d47f7 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-597bf4a627bd7bd2fa8026310dfea3157fb31567dbd0e23b696846cb9f4707d1 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-73139f8e4379c5adbb5939426aeade879e8d72920c0a439f2366f07e95990662 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f67e6c5c | sha256-cc2600bb998788e65195553ee94edd72969fc84a69faa36d119711e9f5fc9779 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f67e6c5c | sha256-a80e3e6ff2a49f2dbc3f24bd32e195cc1e4d4879b03ac20883388c9a27375ace |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d6162ca3 | sha256-34a4d1e7b9a206a59bb1d863d3528e1965f8aa998e3d9cb7fe8e5806420d4e75 |
