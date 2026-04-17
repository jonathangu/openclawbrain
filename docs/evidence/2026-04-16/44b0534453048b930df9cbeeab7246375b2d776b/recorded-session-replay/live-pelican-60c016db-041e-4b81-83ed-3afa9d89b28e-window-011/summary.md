# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f8ed2037bc2ff0feca7432af422fbb2b58a869f3969fbcd41ad42699329bf723`
- fixture hash: `sha256-b54fc3ca6fe17a912f89c0806fd3df709e1f6f80264d7323adb898abecd00677`
- score hash: `sha256-ac96ba8275af238baec329ef47c480063a1aec5f3b48421d42360122d03521c5`
- bundle hash: `sha256-566b1255a6e5d1bcacbf1b6e71da6b54b2cfbb4ab796935c5e60cf9c21dbdf0c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-34738adcfd10c341d46efa61990a3844e1795a8fb18b5ebdc9694342a06a5142 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ee68aa010de03f0f13dbdbbabaf2b1d41d4838f5a72b6f3c9333018a99b0a00a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7b7f6289de25d8d60b6177cb4d9597696dc16da7584925564ba4142b77a2b4f8 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-3d71f37f04f1e8cc14506c023f154cb9debb01d9b72efa87a961376a9e111689 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-423232fa | sha256-6fca5deefd635491ef42c3b5df4da5b33a933adf4e5f520d342edfa821950bf8 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-423232fa | sha256-a9dd53a59eabcddec136705c18deb37800906dcd76f4181a797ac82f543a8b88 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-3d21820b | sha256-855c7487cbaad51d68ba17c50afe107b09c0a4561aadf64f7b735caf84d5a019 |
