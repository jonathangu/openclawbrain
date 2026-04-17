# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b2430ed58ee0abca0aa0224af405db6344da7702ccc6e754dab5dc0867b7727d`
- fixture hash: `sha256-0827a1eef5713f16e574a6c5a2c4721f6c9b9ebfe2794b2f08af42e8c07ece50`
- score hash: `sha256-27ff24a4df099c7e140670a2f48d1db4007a208b4cd183ad0c7dc84bd3f7f872`
- bundle hash: `sha256-f816598329a7d56e2d8de6c77f023e483be7f909fef70514d9d534997bb702a5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5d95dae3d2cb2e3da5df09b63b5296f231dee9a351a91285d6a68ab316bef562 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b91502ad2d2a9f472b5b389684c4e6268bcc9d4ba1598919e85b147f92fcffbe |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4b45cc82e4c504219169410b192b0d6f15d741d35af9211a0761bd6456f343fd |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-0257c2f731339c447c01c8643b920b5ee162eeb6b3d8e8d71b69c44e197db14e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-761e1088 | sha256-649c4cc1969eee1a6dd116223d571a242c45471346c1dba70c98ed4b0113c303 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-761e1088 | sha256-3683241ddeb3a73489f0ed1bae20c625aa0448cec4b40a2d62d127cef61a8037 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-400e838b | sha256-ebfea042a8cd70b5fd56e744e4a0601d932feb9b7406f2bceae29ed0ab32d52f |
