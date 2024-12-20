<!DOCTYPE html>
<html>
<head>
</head>
<body>
<p class="mb-6">
        Qtar is a reinforcement learning agent that learns music theory rules through environmental interaction. The project explores melody generation as a game-like system where an RL agent discovers and internalizes musical principles through rewards and rules.
    </p>

  <h2 class="text-2xl font-bold mt-8 mb-4">Initial Approach</h2>
  <p class="mb-6">
      The first implementation attempted to teach all musical rules simultaneously in a single training phase. However, this approach proved problematic - the agent disproportionately focused on rules that provided immediate, high rewards while struggling to learn patterns that required longer-term planning and more subtle musical understanding.
  </p>

  <h2 class="text-2xl font-bold mt-8 mb-4">Evolution of Training Strategy</h2>
  <p class="mb-6">
      To address these limitations, I experimented with curriculum learning, a common RL technique that progressively introduces complexity. I initially developed a five-phase training curriculum, hypothesizing that this would mirror human music learning and allow the agent to gradually master increasingly sophisticated musical concepts. While this approach showed promise conceptually, it only yielded marginal improvements in melodic coherence.
  </p>

  <h2 class="text-2xl font-bold mt-8 mb-4">Refined Two-Phase Framework</h2>
  <p class="mb-6">
      After evaluating results, I simplified the training to two focused phases:
  </p>
  <ol class="list-decimal pl-6 mb-6">
      <li class="mb-2"><strong>Phase One</strong>: The agent learns fundamental melodic rules and motif structures</li>
      <li class="mb-2"><strong>Phase Two</strong>: Building on these basics, the agent explores complex patterns and long-term musical coherence. This phase incorporates successful motifs from Phase One, rewarding the agent for creating variations that maintain similarity with previously successful patterns.</li>
  </ol>

  <h2 class="text-2xl font-bold mt-8 mb-4">Training with Human Feedback</h2>
  <p class="mb-6">
      The system is structured into two training stages:
  </p>
  <ol class="list-decimal pl-6 mb-6">
      <li class="mb-2"><strong>Pre-training</strong>: Initial learning of musical rules and patterns (phase 1 & 2)</li>
      <li class="mb-2"><strong>RL Fine-tuning</strong>: Training from phase 2, the agent incorporates human feedback collected through a custom UI. These human ratings directly influence the reward function, helping guide the agent toward more natural compositions.</li>
  </ol>

  <h2 class="text-2xl font-bold mt-8 mb-4">Results</h2>
  <p class="mb-6">
      Here are two samples of the agent's melodies. The first is a melody generated from the pretrained weights and the second is a melody generated from the fine-tuned weights, after a few episodes of human feedback. There is not a massive difference, but you can tell that the agent is learning to generate more complex melodies, specifically ones that have more note variety. It also seems to be over-learning syncopation, so I may need to update those rewards.
 </p>
</body>
</html>
