# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-69ce0c4e11baa36853be20e1ca688e734c8855423d37366857eb233deb6e9df0`
- fixture hash: `sha256-c3a333635db8e86be19e8bf48de8cbd13aa6939830c506cedd85267cb0e9f51f`
- score hash: `sha256-c056ade74b9364a5aac68e13c8b05fba4b85c7bafdf62ac5a4e3bd2bb7f16e97`
- bundle hash: `sha256-068a1d08d626d8e219a5caa9fb1965eec731cc8c7f719973e0a26aa8d49e6c4f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-653724b1c50980255f17a34150c96cf9693658619075d0cdd8b7b4b447cb2cb6 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ead5064ff4ba67ad1be831aed147cb1e12a8b34cb612a382da123d92f97a1634 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-806480be40d85f5e0855f4899d6cc275e4c632f7c03cd6b1dd6a26179d15045c |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6cf256f1cec95a4efd3b6228e108a8c57c0c8767a46253e15f04941348dc0500 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b339b6dc | sha256-b73cf344eae89a6bd396dacad8a044ec6aaecea56ba010fe5a1293e957522b07 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b339b6dc | sha256-facce599f8eeaa12dbee422e5d5f0b0b3baaa08474b6841697557e2e16a0c5a1 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-637fee0f | sha256-e643ad362b5f36c04457c2f3d9307df631224086eaee27d47f7e263b0eae0ad2 |
