# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-169`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fa0a3e5a2be78a517ccfe2e1e4b8f4e2529d6e5ac6ae4838bd2c1da5073ae788`
- fixture hash: `sha256-9c907c31d6df545ad3189fb66d2746fb0938842a92e6704858a51c0bdbc6d6a3`
- score hash: `sha256-b5380251dbbfb64c573022fc85b6a47128fbb89a19fa9c072aaee8e8f613a392`
- bundle hash: `sha256-357c7bcb5c112ec6c881c1281bc520779b2d34f080837d5bee697838fe4957bf`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-984a47e062d84a4c2db4727f0e783a355cfd91d65c98b2c3a27a24fa9103cec7 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5672351ca2a0c8adccf07c65592d45f388646102248052d6b32810bdfd25aa8e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d1d1af0c7dfcdcc0c5f221a9c11a3d21e36ca737c0dfcea739f34faecb2fa4dd |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-657330c95750894b2c6186e3babfddcbddb53dc765398cf0abe4adfc1156067f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-750b49c9 | sha256-81c00c56ccc7175a7fe012c90f6f1d1a5a3ccac4adb2548b307f915bb58eb7d3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-750b49c9 | sha256-c23174971b963f434e673b2cca50a19d39bf425200c6131ee7a08be485161273 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-750b49c9 | sha256-81c00c56ccc7175a7fe012c90f6f1d1a5a3ccac4adb2548b307f915bb58eb7d3 |
