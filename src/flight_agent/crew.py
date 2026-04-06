from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from flight_agent.tools.custom_tool import TAVILY_MCP_URL, apify_flights_tool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class FlightAgent():
    """FlightAgent crew"""

    agents: list[BaseAgent]
    tasks: list[Task]
    
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def flight_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['flight_manager'], # type: ignore[index]
            allow_delegation=True,
            verbose=True
        )

    @agent
    def flight_scraper(self) -> Agent:
        return Agent(
            config=self.agents_config['flight_scraper'], # type: ignore[index]
            allow_delegation=False,
            TOOLS=[apify_flights_tool],
            verbose=True
        )
        
    @agent
    def web_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['web_researcher'], # type: ignore[index]
            allow_delegation=False,
            MCPs=[TAVILY_MCP_URL],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def scrape_flights_task(self) -> Task:
        return Task(
            config=self.tasks_config['scrape_flights_task'], # type: ignore[index]
        )

    @task
    def web_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['web_research_task'], # type: ignore[index]
        )
        
    @task
    def synthesise_recommendations_task(self) -> Task:
        return Task(
            config=self.tasks_config['synthesise_recommendations_task'], # type: ignore[index]
            output_file='recommendations.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the FlightAg crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=[self.flight_scraper(), self.web_researcher()], # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.hierarchical,
            manager_agent = self.flight_manager(),
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
