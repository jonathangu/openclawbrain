# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2475fe2a6c8f093359c7a9114559d8796f41e9f7cd1ec07db7813ffabc8da40c`
- fixture hash: `sha256-45ef9df8d6b446f8b1f701f41e5b1b2fc10fabaee25942a7631b331568aa14af`
- score hash: `sha256-b38459c0913c321d721ceca4c88fd112c86805e4234126c3855f37ef7b9aaf5f`
- bundle hash: `sha256-bf802c2cb2e599d74e81c82c08b13732b3303b157aa05d833e21a0264228ba32`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-18e80a3b5ca91ac2a6f8025eacf958282c07c535eea3cd785fd5adb98dc2f9c4 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-cfcb0628518d8716ab2953c42cfa3a222fd442eee90a4d759634544b75e01316 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-8e542cfb239bc2a167a22935fefab8cb3d28faca889e9051dedcbc40b08c2480 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-84359f7428aabf7cc04a4b829b8a3cd5f1d27ce7ad47a513dae6c0b19a71d529 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-6b05aaae | sha256-872a76d817d0a491fa52074dd76c4e6baf10199149ff02195784fecefefbe663 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-6b05aaae | sha256-be65b43f18bc929173911cad386b25b1362a4816c7cc6daea334448a6514ff04 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-6b05aaae | sha256-872a76d817d0a491fa52074dd76c4e6baf10199149ff02195784fecefefbe663 |
