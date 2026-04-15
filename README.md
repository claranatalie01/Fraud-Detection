### Note
This repo contains relevant works for sharing among our group. This is not yet the working repo. Let's also use this to update each other and keep track of our team's progress.

### Goal: 
A Fraud Scoring version of the following solution: https://pasa-e7fad.web.app/agent
- click for the ***Demo*** to view full solution 

## Individual Progress
- **Clara**: created retriever agent pipeline for retrieving incidence of fraud for single transactions. Need to design how output features are to be displayed.
- **Julia**: studied input-process-output pipeline for risk-scoring an account after fraud scores have been computed by Kin's algorithm. Need to work with Kin to patch output of Fraud detection agent as input to Risk Scoring agent.
- **Kin**: trained and deployed Fraud Detection (inference) agent for structured (tabular data). Need to create data processing scripts for (i) Bank Account fraud tabular dataset and (ii) HKMA JSON dataset and train inference model.

## Timeline 
- Wed: Kin creates preprocessing scripts. Julia implements whole architecture of Risk Scoring Agent. Clara drafts front-end design for retriever agent output, workflows, and how agents connect to each other. 
- Thurs: Kin trains model on (i) data and patches output of inference model with Julia's risk scoring agent. 
- Fri: Kin starts creating the front-end until Sunday. Julia tests the fraud-detection + risk-scoring agent pair on different test datasets. Clara finalizes the workflow design.
- Sat: Julia continues to test the agent pair to produce statistical results (automatically done by inference model). Clara starts working on the PPT contents.
- Sun: 
- Mon: Clara and Megan work on the PPT. Julia helps report the statistical results and findings of the inference machines. Kin finishes the software prototype.
- Tue: