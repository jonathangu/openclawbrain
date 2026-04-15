# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048`
- winner mode: `learned_route`
- trace hash: `sha256-9c32a87b231e4d5848a772d9d1cb8d355e8b17c5c883fc0f1ca8776ef042ba2c`
- fixture hash: `sha256-66d4441e9cd89d5df06e129fcf70accf27e8123573950bf81a6f813e2979adc4`
- score hash: `sha256-837903664ed6ceb7a47a73a180f4c697866124fe0aa159e65183433e09b49c2f`
- bundle hash: `sha256-f84dc11137b33ca2b153317961a31cf34f80f5025eaed4f427dfb3d5c4b3d7fa`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b9d9843197c6ea9cf1bbaf94c65647f4ecfa1e2224f8678711a552cc896cd7e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-d9de9093f1fe5991a494a10fcbfdbb094b732b9c397161e8d8564c7770db789d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c3a1cb5994f8020a2ff6ed3e2e66be7dc17ee53452750aace942b4d059cb14aa |
| learned_route | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 2 | sha256-3cbf6782846e7053d8828892626ac970b61f704324fdb4ff512e4878480a8de2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-6b671760 | sha256-8a60d812725f7e0d0a5a0af2e4c27f3c2e75d884484c819fa1538b3709652b3a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6b671760 | sha256-4b2acba8de794eb16ecbbe944277e9bcebdcfce5c96dc8495f87ea00f91c93d0 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | no | pack-6b671760 | sha256-8a60d812725f7e0d0a5a0af2e4c27f3c2e75d884484c819fa1538b3709652b3a |
