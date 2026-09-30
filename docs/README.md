# Pipeline docs

Read in order: [concepts.md](concepts.md) (the data-engineering vocabulary), [discovery.md](discovery.md) (what the upstream source actually contains and the decisions it forced), [schema.md](schema.md) (field-by-field contract, bronze and silver table definitions), [stages.md](stages.md) (the stage plan, done versus planned). Decisions that shaped the project are recorded in [adr/](adr/).

[evals.md](evals.md) is the agent's golden question set: what it asserts, how to run it with and without a provider key, and why it is weekly rather than on every pull request.

[agent-service.md](agent-service.md) is the deployed serving application: the FastAPI app as a Lambda container function behind a function URL, what it reads out of the lake, and what a cold start costs.

[demo.md](demo.md) is the five-minute walk-through: the commands that take a fresh clone to a queryable lake, in order, with what each one proves.

[orchestration-on-aws.md](orchestration-on-aws.md) is the Step Functions design that replaced the AWS stretch goal: not built, and why.
