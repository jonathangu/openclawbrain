# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-082`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bda6b3da4ef39b29be45310328eb0566a39d316769663e62675a6105dd7880f7`
- fixture hash: `sha256-e12b530a582d1487040cb7cdaf3e1255576e9298c334dbf79363d1f81080b1c8`
- score hash: `sha256-28412011345fe15cafa6817a074c4d59527658567a857faf99a5218b447cfdf7`
- bundle hash: `sha256-d7f0250028a176abab75c73468f2482226849494a205bca664f9114ef5aac76f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bab067ff9232cc412579013f9b35dc498686eb53e7f83b8de58e12e80ba3c742 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c19bd76bdb7fe64388f80a415360d264be6bca2507d10ad91dee896617d62a15 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d552e3a37ba334e3baa25b48a1a69feae0756bdf9b35633e84c0bb5215b5bcef |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3b5b279556b26a0cc2ac5a9e8e8594531950ff2b5f56058ef603900a744d8ef8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8e63899b | sha256-b06eb59c506d3bdd7273f93219e7cae48d366ca6dba5348a448fec86a1d7a261 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8e63899b | sha256-74068bbb3df4c00804d7551b734ffb42abd6706fe30aceb7fbf3d406935fd8eb |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7c75deba | sha256-a11d7734acff564cc5e0daec5f289f63e738e15263920873df8d3b0e0039e7f8 |
