# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-051`
- winner mode: `graph_prior_only`
- trace hash: `sha256-00bf2bd686f7cfc027d3b7749683ef5ae4ebe1a8b4b5f12763771285b87ec8ab`
- fixture hash: `sha256-1287af06cb4b83146712d66b78f07ce6e6ac74450d156f3cf86e05b95cfe0f1f`
- score hash: `sha256-2580ffa25d592dc8cf8cbc8bba306f21db1c722b867648c54555574e44e8b843`
- bundle hash: `sha256-43f76680a8748353e66235e81e5fa5ab1de83b41dda9d482ee11cc6509ac11c8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80a7f9806b34ab4aca7f2c918d805e0ef978c8cb5147a44aad086817dfd7315e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e88a2307548c08d131f69dfd6d3e8f99ebfdc10719573503f7907aa10d24aa29 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d55a58c6407687598cc1634c397a3aceb73d0affa09de4b372cd111d68917dd3 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-0a892cef18d6714b79aa92c75fff672a76e03e7941560256c5cdc05255560e75 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-04b99e1d | sha256-1cbf2e128f8ddf981ed80a246499f27e90cd452738bce68efbaf23f7f270e9dd |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-04b99e1d | sha256-f3e0ef5d52f43b4478703a441391a16ff3ea0085312a675cf95b1d5197eb7f58 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e429b766 | sha256-3e799274756c16ec68461b100f17212bec628ef47d5a4cd80130caffed010f6b |
