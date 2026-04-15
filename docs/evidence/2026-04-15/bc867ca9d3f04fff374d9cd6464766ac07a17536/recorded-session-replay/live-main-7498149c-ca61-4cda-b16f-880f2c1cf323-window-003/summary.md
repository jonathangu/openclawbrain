# Recorded Session Replay Proof Bundle

- trace id: `live-main-7498149c-ca61-4cda-b16f-880f2c1cf323-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e725d27535b9af2607edfc55a55297b3d0ba2750ec3f143ed7ff11bf77f51432`
- fixture hash: `sha256-ad7016b7c1b3014d2314a0a7c149b979d45fc36075f6fdac2c19edc868ece1f9`
- score hash: `sha256-5a10f6a02df4711cf047b72e5ac9ae5e6031dcf849105b35f178ac44ca82cbbc`
- bundle hash: `sha256-28ea7f8c5a646c8432f459d4092dc5a73c9b6a80fe2a5e0b83d643d59b6e29b2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-876d8e0d55a1dbf8e84f9354102b290654bd4fd9a97c3f26b37887de51274c7c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0655fb96ef4e81c7ba34b17281e28faa2426e7d4cab85fc85ce757a716e7801b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8757e8d01f81cb2171f2492bf12adcdd60f0e5b26b826e267cdcd100d486a525 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-45c36315956a4d0b905f9fda39b54adf3311bb59a8532798df3cb6bf838eee83 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ab19168b | sha256-a9827c42e35003ba66ddeeb422426a7ede90fbc218bb9ef10096b89c263b6ef1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ab19168b | sha256-002a20a7651ba270007567198f1c9f0b05fc6cb7bd324b79b70424b0e53598bd |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ab19168b | sha256-a9827c42e35003ba66ddeeb422426a7ede90fbc218bb9ef10096b89c263b6ef1 |
