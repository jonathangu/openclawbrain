# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3a175359354be687e420ebf0f5b7a7ba6d56063e99f8d68c1b9194462319f20f`
- fixture hash: `sha256-598383fefbd171f64af80d75b4d0910cbb4c4236c56c222c3e6f677cd87ecd08`
- score hash: `sha256-c22509062696f8894c50f4a0e3f0480e522e24ae579725b8d961683eef83f641`
- bundle hash: `sha256-ce0c2ae709598c284ca7e2bf5f45e964033e0fad29d871c723df105de83caa44`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-566dca0adc9e80f718fe72c8b1c0886a7b904a433ed25544bd407de590aa9335 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-ae6b84081a2c05ea9ac76f8b06e343d8eec91149f65e19731d7ad0692912cf29 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-6eea4800c4922d531f41652258ea1ba86e057559b2c78039b107130166ddc6f8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3db94c8830ffb7b883df383fa1cdb4fb8a11d2eaed466f7b40b6e0fd8d993a5f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-c432b1fe | sha256-9d124cb9b8c24c86936b8effcbae481cf9d189508885121ceca4ba7b5d738c85 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-c432b1fe | sha256-b4565c1bddf208500ce84f46d16e73702b609b5bab5d890beab81eadf2e6d6b1 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-c432b1fe | sha256-6512f8abbb3a0fbe9953b19d14c5f65ada6add8509e7808f51e2b0a7391152dd |
