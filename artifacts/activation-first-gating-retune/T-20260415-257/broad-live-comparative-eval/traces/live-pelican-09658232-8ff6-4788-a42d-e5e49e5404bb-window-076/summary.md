# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e90ba91fa2d821b34e7d50d49031d2ca2e725469eba7413ed1eefcf887d0f975`
- fixture hash: `sha256-b66c57ad146f945a1113822081ae1bceec873a0abb858cfb6bafe580d07b22c8`
- score hash: `sha256-b2ca26ce5e0509459cb7c238017e391a06b41a3e071576b3e9623683165e9f53`
- bundle hash: `sha256-1a0d319139c809c6c71564d8070287dedafabf85dd9bba1cf1e24b785b90a188`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d7a8f01e83ed8ac33586c073703951c8627b99bf4e9aa0272b865992ce2738f9 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6b864141b2ea5361ce2df508a19aad79389dffd24d7385565de41ca65cc4a9eb |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-455145d982003048099e1edcd04e87b08f256c5b661f30905cdca22f980e6edd |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-5d213ded41e38a93558daa55baf32fc5f3ae98cc24b7364ea1bc0d634efb7f06 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-72202536 | sha256-d99002fb981cab83a636ca5b5a1039275f4ff9726da14974906c395531f4010f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-72202536 | sha256-c825236c3c37aea6d44e3cb2622f2ba412143111f3ec47f75bed3cd2a3e71da3 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-965f0ceb | sha256-ce785af007f2e2c25c27038d59189395db1eab3960a9a12d1f1a3b9f6d355b0e |
