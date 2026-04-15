# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7ce9d77f8d5b34f5d4a2ff238035837b9a17936c8718ddcd44e0135af5ed67b2`
- fixture hash: `sha256-b278e1b6b555771ff403bacda1c9f56aa4593110af14f3b45502af98316b55cf`
- score hash: `sha256-e687f1cb445a38f7d611d36ad0ea84c0bd9f54314d6669e2cac7aef44e4fa802`
- bundle hash: `sha256-0883e59994718bacc926bf5e08b11bb96cd6253ed67109dda5f7181a4ece420a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1fde92dc9f75c3ed7b5cdbc92af57a8fdea90f988cee9df5a6592eb109fc517c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7d4423577c067100821055c9e258e9ad2d3abed8e77c1592558826bc8df337f8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0182cd78ac68b4d32a20926e78e943a4501ffd13a0eac83a12c0c3defd28f9ca |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f001fd6b1a4da823b4083bc6633f0015ec8d2f8f13a3467ec8e53bbac9163086 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cc456773 | sha256-9a14abb9a94a26155fe3a5695658a428a1e8040b7e76cdec8a7d7459af2d61c6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cc456773 | sha256-97e88881fbf35a0bccf1f91a58112da4a8584b67b4eaf4a519faa89ce8c572db |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cc456773 | sha256-9a14abb9a94a26155fe3a5695658a428a1e8040b7e76cdec8a7d7459af2d61c6 |
