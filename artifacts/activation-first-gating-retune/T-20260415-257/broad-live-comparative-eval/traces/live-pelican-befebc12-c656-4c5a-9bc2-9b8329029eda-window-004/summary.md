# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-974c0caac77d24b74750a03083f2fe960327dbab94f044b1f352645b0c8977ef`
- fixture hash: `sha256-5332e2f75d9b84ce32dc4225385441cf4e1fdff1345733b1234e8eeb65449d9c`
- score hash: `sha256-880dff546c960620ed690e7df7c7b62d7e0ce0aff47f15871f07f51675d06133`
- bundle hash: `sha256-40627afb39097d5cd865d1074add28db978d88474a613de7257a4b2baf577487`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-dbc3f90d1fd6323456b57dfe2268d1b7eda59d985039bca3aa485da45edbfc64 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-59b8a01bfe6beb04ebe4780eeb1ae5ffc511be4669c950de5d2d1e38d901956f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a4ff070b7e1008ee7831160b8dd181a46000f583dd3d71780ba9d026f29a32bc |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-75fa99d27ae7ad0844b93ab9284e3f86151ae560cbd1789ccd23b396a6d58d99 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4cd99d07 | sha256-992aeded1d8499cd7a84b7868bd189d03c59ee4f9f209120a439dae553f97cb0 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4cd99d07 | sha256-6fe39b8c9a0477231b7f5569f78ec261c9e333667ffe67e896836aca19ccd1b4 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-7cb20e84 | sha256-472529b53af158a98cd7301348d293ac9ce7dd0ab2fecbb3231bd46bc5a4f5a6 |
