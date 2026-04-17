# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6cd785628fb3c34642dd7b4a701799a6e96acb06e347a7bf1d01cd4950a8de4c`
- fixture hash: `sha256-8169ace4aebbc5a4a546b5c0d2bdc7c5a395f1f1630a066be79c7f63594673d2`
- score hash: `sha256-fdc5c743b8a7e8429d928a504df9a92c80e3fba02a10bf4b95adf712af1553ad`
- bundle hash: `sha256-52f5abe10c6872361e52f689fe3e94f3958025fd22fdd1d6b22b4ceb08a74340`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf6fbe7613b07ee3e659c5ab0ce2fe9e83640dc0dbe17b255f0c268784354a36 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-8d3ddb5bb303cc4b652fb3492784a11657f23a33075990d5e09655d73b6aaf84 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-2429ee6ca079fddb039407e15198684250fe23b32797db49bbd1f9384cd2ab04 |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-83824f3fe7c43a15941588dd4115b37f7b199ebe03979158507c16bc6076b4ec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-86dc2a64 | sha256-d49046a4ddfcb5700f458a167f70cd00ea2fd8281ef7b1bdab57f82aa96c1ca0 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-86dc2a64 | sha256-45dfe850a99adbdf631576edbe0802cb499589ba523aa62bef6f79dd4a6404e6 |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-3d64442b | sha256-73d359deb3e356c317f321f01661056af85bc97efa3031c16f0418916e0c5a20 |
