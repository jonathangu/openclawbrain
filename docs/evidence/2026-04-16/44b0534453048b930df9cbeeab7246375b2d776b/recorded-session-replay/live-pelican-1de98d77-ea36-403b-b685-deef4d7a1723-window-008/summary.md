# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-59b36a090be7bc29212f1b29aa7bc29b12f23e5a450aaf05d1c3eab4e44abc8c`
- fixture hash: `sha256-712180e16240a5850bc7f0f166cbbaa035f07312fa10c3e98606123034cbbf4c`
- score hash: `sha256-795d14bdb56fcc87282e5be65573b525e5977b687197c5e4ee3ab6e6a941ac4e`
- bundle hash: `sha256-4d6fcfd8f0ce79d470b5bb2181b32964bd585cd48e1df185bcbc830773969b54`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b108d912f44bc4b15526ab7db40ee75e868952f9ef4952b3a83ae96ae65d4c1 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f50b386ddac6960f1addc8226b572f567fa0dbc661b3fbc4476bd6de50f7079 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-a112252cafeaf526816ede11bde17e95313d4fa734889aef8572ff482051b6e0 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-94b1b1f42827e2e2a48483620f60f423b73ee929ba095925f72d020c078e30b4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-bef80c09 | sha256-72ac882bc8c2f607388a6ac453d6b19f616a143ba12a2a6e6581db4fcf8dfeb3 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-bef80c09 | sha256-e6d101c82c66fbe0cd035fcb0db1038574b5084682fa39f04155be087b4d6a27 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-5ea8dd10 | sha256-d6ac2231a9e97b926aca34c5dde1f2a775d58f41e1cde4f4e85920d517a0a249 |
