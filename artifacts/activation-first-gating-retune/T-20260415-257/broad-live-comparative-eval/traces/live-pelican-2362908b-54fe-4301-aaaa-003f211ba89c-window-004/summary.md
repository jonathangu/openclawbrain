# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae1abddec00632179423e5d665c773fa81ea75d92b306fc15251840d9f53ec48`
- fixture hash: `sha256-c2c90149661c99c58bd2b000a17d70b99f16ed3daba941c64a7e5c1b67ab99b9`
- score hash: `sha256-fb10a6dcc60306f3f411ebf0aa4c8a9a8119d667901cbad34a8c3319c64fc504`
- bundle hash: `sha256-049f0f5e1f48b045ef0954288eb7f68e48015774414dd5572f265b6c7cc49ac3`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-83c42e700005538dba5b3a6d69c6c5e443ab91af8b598837eb4ca6b5f8135237 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0cb0a6936df35908b4514c445c98217c134511f6371d00029cee85cd0ebba27c |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fcd61639c1c356b1706d915f238a37a7512dba1d6d8b58ea4163a70b0bb0ad30 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-9dd3acb2dcf64dcb83974be92ab7940dadce4e7677d92485de7686511eb85233 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8552e430 | sha256-7ca8120a2d9876e2ad7114cb58b9cf4e6fa722a5d49e623bb27496ae095db8c2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-8552e430 | sha256-9ece58150ac852af6ff6ea30b2976749d84ae797368a3ec5f223a8e0427fcee3 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-e66396a1 | sha256-478fdc97758d4f948cd749fd476bee17fa6cec41c49f3ad94f27d4174a0417da |
