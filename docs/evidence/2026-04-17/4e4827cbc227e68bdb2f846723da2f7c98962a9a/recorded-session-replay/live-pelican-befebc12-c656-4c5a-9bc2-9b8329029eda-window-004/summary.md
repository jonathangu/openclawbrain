# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-974c0caac77d24b74750a03083f2fe960327dbab94f044b1f352645b0c8977ef`
- fixture hash: `sha256-5332e2f75d9b84ce32dc4225385441cf4e1fdff1345733b1234e8eeb65449d9c`
- score hash: `sha256-0f639c2ff6aa021918a1eb5291c706e0b0e98c454eaa25aec65d106797f69841`
- bundle hash: `sha256-fee0b8b7cf5ba2581d030045d24973739fefadfe530017610dfb5772e095e2ff`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2a33d3a5e556a2ec0cfa921af4c9a61d0c77e11f2fa99dad9935585d31de496c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-eabe515dfe41011622d8f3edada9f8116567df48a96a5d47bd385546e9a1ac4e |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-bd2ecd08e064261d7af3601b040ebf15a5e50c3dc65ecd87961fb0707b4ae9b8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b0395e68 | sha256-789615616d09ea16160529500e10e82598513eaecf1b406ae82158fd8ce3756f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b0395e68 | sha256-b1f576d7ea42ab014ecd8e072c9f0c01660adcd27875cc592204d66c43f4ce5f |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-e011cfe5 | sha256-443510df1d66331a12a7dd0f5266cbe77ab38679472a3805374022711b47ab6c |
