**This is the initial version of this work. The author plans to refactor the code and release the final version of the code before July 15.**



## Installation

LLMArena is developed based on the OpenRL framework. Therefore, you need to use LLMArena with the following command:

~~~shell
pip install -e .
~~~

## Usage

First, for closed-source models, such as ChatGPT, you need to use the methods called by the API to experiment. For closed source models, you need to manually encapsulate the model into Openai API form

~~~shell
export OPENAI_API_KEY=<Your API key here>
~~~

Then,

~~~shell
cd examples\selfplay\opponent_templates
~~~

and create a folder named LLM to be evaluated that contains opportunity.py and info.json under each environment to be evaluated.

Finally,

~~~shell
cd \examples\arena
~~~

and replace lines 123 and 174 of run_arena.py with the model and environment to be evaluated, and then

~~~shell
python run_arena.py
~~~

