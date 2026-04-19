# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eebf5a156737b8a1b13583833520fd34225ae0f30b4afb05ce10671f54ea2108`
- fixture hash: `sha256-63d1e3dd69e143127b58a78b17f85b8f588fdddd25950ce30e59877032c4d44a`
- score hash: `sha256-3c665e9ecb27dd10fa7234686186f45c872737258f8d16eb1b7a9a448ed2c4f8`
- bundle hash: `sha256-d5f13942ef43fed4aca73b8bb667f514d55b6f224fa12fa05b94908ffdca45d8`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-798f7833fec1b062f8a2789c97b9a979ee4a90e5e78bf32289929e17459a82fb |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-5ea5261e08ac89c057f603782cd21086514bbb9cdf25a34de3d9d748811369ee |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-a30758b53f9194191ce56d456e7922d582260be3727a59428eab3bd31626f6a8 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f48de3e1aa5291aaae6cc11798d7ad01e69ddf5d9e3422ca924951822aa40395 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-d666beb4 | sha256-d24bff675cad7ea9f0a4f6331298df02971662d424032ec0b5686eea15e45b27 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-d666beb4 | sha256-f2fd9c317509cec1fa495709f48d7cf3fbd9096c4f1772b689697140efea295e |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-d666beb4 | sha256-d24bff675cad7ea9f0a4f6331298df02971662d424032ec0b5686eea15e45b27 |
