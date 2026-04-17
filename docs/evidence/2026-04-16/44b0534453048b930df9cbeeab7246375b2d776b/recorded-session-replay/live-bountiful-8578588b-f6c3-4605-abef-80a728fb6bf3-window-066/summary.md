# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-066`
- winner mode: `graph_prior_only`
- trace hash: `sha256-63fb4cb45ad9dc0c567733233e10232470bfdf8ceab21d6358838e826794486f`
- fixture hash: `sha256-9867351ff6099d4ad5c5968b4a566b26ecb9aec41e3d4c81142c7386e19d8bf3`
- score hash: `sha256-b462154e801b9b4d2aa3a747535a6eab54e43b551222e695c97374511b24e596`
- bundle hash: `sha256-2ff95c791544d937c482a2785d8c3e3a4aae781b9642f1a1eab3f6c943f95dc1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f1e62d93a8b084333b11d3b582af82ded5069f848e9e733cc9f3aff54320eb3 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86532998e62437f4cc6aec93bcfebe2ec4ef1f39f2af7ada6189583fd39a0760 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aa13a9dd76c6ba15feafbaf3284389e476258f69ad6b4689f2425837664cff7c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-239952d8299581d9e28bcacf3b3690316902cccf943bc6b38721eca3ccb9b32a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6b6a25c2 | sha256-de48bf069eb823ef09f534b1c06cb77db84255f40edb35004d6a053dd272bd67 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6b6a25c2 | sha256-9e9f12abe0d1a34923f6c4b590571ba13e2c2f06811fb95e356469550ff39c2d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2c7d5b1b | sha256-a19231f48b31efe535d95b70716d62bb764be0440f730cee7b22b547514ea134 |
