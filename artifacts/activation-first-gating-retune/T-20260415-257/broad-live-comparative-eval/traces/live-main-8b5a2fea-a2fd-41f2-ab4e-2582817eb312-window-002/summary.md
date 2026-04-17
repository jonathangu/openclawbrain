# Recorded Session Replay Proof Bundle

- trace id: `live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002`
- winner mode: `vector_only`
- trace hash: `sha256-e0e56ffd1c26d20085e7a9eb3248f58dfab8c43d92d6bc35e804da203ef4f7d9`
- fixture hash: `sha256-e4b8d39277cb985d3e9ee559f9e373775182720bfc10b6d9350141f9c5016460`
- score hash: `sha256-6a65ea4faf62399d761b5c5137c1f043f1ad65249a3c77943ea08ac7d2435471`
- bundle hash: `sha256-a2e80a62303f4f43d73dcf963e96fb88a393662758ba48b1f27050c94c40a8d4`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 80 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0bdf6c0bfdc77dfb35df2ddd80b080b8e6bbd2f8f1020fedbea4770e769e1c72 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-1151ebb0713cd3ffdd7cf6395dd71d87f05e94b52230d2c5ad512dca62e98015 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-93e805288bf70a5e70758e396fb050b3d15fb45838b4048290156d2a3f43c6c6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f25a311b35a7a2f24b9f1657a977a539386f6f18d353f4c98c1cfc0dca47a87e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-2e008ab8 | sha256-bc41163b38c929b87165a429a0ea340e505feb1b16868b06f8cee413338a2e48 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2e008ab8 | sha256-9697a89a17ded2ad90e8a59d3e713f73ccc94a3a38e8ec46d5effd9eed6c0839 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-942b5171 | sha256-ee926e05bdfe8db9bb1274affd7285e5fb62cf15ad59ce994f1bddb2fa0cae6a |
