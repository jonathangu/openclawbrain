# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-024`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b224dc602b7429463a9b2fd5346afa6d3382bb3fd84bc9d3cceb0d3ff24896dc`
- fixture hash: `sha256-493fd471e0bb608979cd024ca51b9104b86ec7063e95845a4d6e7076002d21f4`
- score hash: `sha256-5c9e2416d2c506755c390e1b43677e1166a7088e1855599820846122ad35fadd`
- bundle hash: `sha256-467fabece52a7dd3b4def2eaae1763f834270303e80b9a4d8b82fcc40fdff757`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff276f984ca7449fbf40ed52f8c73e2aedf05be900e45cdc0a8a0b8a46668591 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f746c40dba4e6eed0049a31f8661943ff3cb269f23a3e7fe242a8e2df09be57 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0873e944ce86c4622c125a8c3cafc0f040cff6899b4fe59836deeac6adcc7356 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-82f0078ee84b69f77ecdba289462ec3fde53d1aff59cfdcd04464b0e30541662 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9c8e3dd6 | sha256-ffe740905efb8f7ad10c3c9c8dbe17596a52f7b26695ebaa7fc24889b5f27803 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9c8e3dd6 | sha256-a1ff7c69f721d0a169509766e9070373a44ad7bccdcd80df8b89f529c8cb809b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ada6cc7b | sha256-787d43714250a4bdca841c975a678ab134926b7c2bf729163f79ff0191da8ea0 |
