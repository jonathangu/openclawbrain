# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a8a2e6a63cc5912fb58030e76267c771c6d07671775935e13384022cf8e7c59`
- fixture hash: `sha256-d3b9199b3d1fba06ec6d727611496f93d92d13e1e28ef25defc3314d0f80c421`
- score hash: `sha256-636763ddcb191e78fb9e744c9498cc7b1c6da63f5c5e728fa6db36b526cd9273`
- bundle hash: `sha256-da0ef73a2b114b081203e238715b20b809590cb0a18c224bd5e018a292715583`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | vector_only | 70 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-39cca038bdbd32b11125d0c6fba3b1b3a673e66a982ba05e8a320b541d748401 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-0710534667364396ad151a0ec49a5f092b5d4290101df8f8f516da762c6913b2 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-52daad826d09ff6b5bf44bb360d4f58dd28d3efb33e385e94e7d01413620a9b9 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-5a1dc306cd9825bca223e0d863ab00dc2da457a815bbea7656cc72d1146efda7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-7e575b6f | sha256-231b7667990fc69e8435ede325fd76e2690336a5c6a6df7c297edd7798a9fbfa |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-7e575b6f | sha256-fe4632bc9a6b0659abf5f2e8e42fa4e1705688d725c4f6542d2c5b170343ff9d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-647a717c | sha256-75a54f08bc39954a682187419c95fc0e4652eac699591675f29aafaa7d5c98cf |
