# Stock Technical Analysis

#### Setup

Few points to remember

* If possible use the git branching model according to [this](http://jeffkreeftmeijer.com/2010/why-arent-you-using-git-flow/)
  * There is a [git plugin](https://github.com/nvie/gitflow) for the same.
  * It is important that we branch from the branch develop (always), not doing so could cause merge conflicts.
  * Only tested code would be pushed to master.
  * When merging features to develop please create a pull request and then merge. This will help us in roling back the changes.
* For first time git users, please use a git GUI like sourcetree. 
* Project management: https://trello.com/b/sXpA1eym/stock-patterns

#### Installation

``` pip install -r requirements.txt ```

#### Execution
* Data collection
  ``` python main.py 1 ```
* Pattern recognition
  ``` python main.py 2 ```
  
***





MSFT PIP_output PIP_times
e.g. MSFT 23;12;234;51;20;35;12 11;23;30;...
