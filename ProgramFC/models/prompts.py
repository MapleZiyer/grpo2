HOVER_PROGRAM_FC = '''Generate a Python-like program that describes the reasoning steps required to verify the claim step-by-step. 

You can use three functions in the program:
1. `Question()`: To answer a specific question.
2. `Verify()`: To verify a single fact or claim.
3. `Predict()`: To predict the veracity label based on verified facts.

Rules:
- Each program must only answer the specific claim provided.
- Avoid any repeated or redundant lines in the output.
- The program must be concise and contain only the necessary steps to verify the claim.
- Stop after providing the required program. Do not generate additional programs or repeat the process.

**Examples**:
# Example 1
# The claim is that Howard University Hospital and Providence Hospital are both located in Washington, D.C.
def program():
    fact_1 = Verify("Howard University Hospital is located in Washington, D.C.")
    fact_2 = Verify("Providence Hospital is located in Washington, D.C.")
    label = Predict(fact_1 and fact_2)
# Example 2
# The claim is that WWE Super Tuesday took place at an arena that currently goes by the name TD Garden.
def program():
    answer_1 = Question("Which arena the WWE Super Tuesday took place?")
    fact_1 = Verify(f"{answer_1} currently goes by the name TD Garden.")
    label = Predict(fact_1)
# Example 3
# The claim is that Talking Heads, an American rock band that was "one of the most critically acclaimed bands of the 80's" is featured in KSPN's AAA format.
def program():
    fact_1 = Verify("Talking Heads is an American rock band that was 'one of the most critically acclaimed bands of the 80's'.")
    fact_2 = Verify("Talking Heads is featured in KSPN's AAA format.")
    label = Predict(fact_1 and fact_2)
# Example 4
# The claim is that An IndyCar race driver drove a Formula 1 car designed by Peter McCool during the 2007 Formula One season.
def program():
    answer_1 = Question("Which Formula 1 car was designed by Peter McCool during the 2007 Formula One season?")
    fact_1 = Verify(f"An IndyCar race driver drove the car {answer_1}.")
    label = Predict(fact_1)
# Example 5
# The claim is that Gina Bramhill was born in a village. The 2011 population of the area that includes this village was 167,446.
def program():
    answer_1 = Question("Which village was Gina Bramhill born in?")
    fact_1 = Verify(f"The 2011 population of the area that includes {answer_1} was 167,446.")
    label = Predict(fact_1)
# Example 6
# The claim is that Don Ashley Turlington graduated from Saint Joseph's College, a private Catholic liberal arts college in Standish.
def program():
    fact_1 = Verify("Saint Joseph's College is a private Catholic liberal arts college is located in Standish.")
    fact_2 = Verify(f"Don Ashley Turlington graduated from Saint Joseph's College.")
    label = Predict(fact_1 and fact_2)
# Example 7
# The claim is that Gael and Fitness are not published in the same country.
def program():
    answer_1 = Question("Which country was Gael published in?")
    answer_2 = Question("Which country was Fitness published in?")
    fact_1 = Verify(f"{answer_1} and {answer_2} are not the same country.")
    label = Predict(fact_1)
# Example 8
# The claim is that Blackstar is the name of the album released by David Bowie that was recorded in secret.
def program():
    fact_1 = Verify("David Bowie released an album called Blackstar.")
    fact_2 = Verify("David Bowie recorded an album in secret.")
    label = Predict(fact_1 and fact_2)
    
# New Question:
# The claim is that [[CLAIM]]
def program():'''



class Prompt_Loader:
    def __init__(self) -> None:
        self.hover_program_fc = HOVER_PROGRAM_FC

    def prompt_construction(self, claim):
        template = None
        template = self.hover_program_fc   
        return template.replace('[[CLAIM]]', claim)