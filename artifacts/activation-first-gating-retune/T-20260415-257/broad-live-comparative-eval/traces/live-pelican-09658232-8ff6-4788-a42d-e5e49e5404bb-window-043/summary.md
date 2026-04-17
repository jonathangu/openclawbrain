# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-043`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5dd6be200875e55287b6b027aec6adb1211d2b987b83c6a37b985516f7118529`
- fixture hash: `sha256-eec602e66445ff4dd47c7240e799fd3d8564ee87f3fa97f5e6b5673abf356c14`
- score hash: `sha256-1ca570eddee6ab82fbced8cefe10534a20a40d20b32f3f9746bd54811dbb974b`
- bundle hash: `sha256-e03d4aa834383c9f06587e2f5edd8206773b570d4fdace1017f2d15493ded0c8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d99ddcb432bf41d8fa10f8ab6904c40f835adbab6565ab293b9f4c7f5ab02130 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-648ff5e47a4418ac0ad08968aab0b690238c804690bf4fd9f0ce733057c46929 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9a0ec65d292969dac7d7223aea0c79fa903588cbb42e2462a471760f9b8a0dfe |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-c60d9eae1531bea01e680788ab95f953c8b17297c4b5f72755249f09c4bb0f6f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8457676d | sha256-63e1e85456ca09a9605eaffd922ca8afed6e925c09a95a49ed7d07c9b74b85a3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8457676d | sha256-a5f9500b2c238c5bd4b2a21246fd3597811c43ce7491035ec50d1a90fd275b7c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-897dd88e | sha256-8f00edf18633fde13394e39d62df2c252b88fcc77ab8ac4e3556bf546f8cde3c |
