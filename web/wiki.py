import wikipedia
import re
from utils import Similarity

def wiki_search(user, query, full=False, top=5):
  """
  searches wikipedia and provides the results. 
  INPUTS:
  user is the user's question
  query is what the language model searches for
  full if i want to return the whole wiki page text
  top is how many sentences i want, 5 is decent
  OUTPUT:
  returns the text thats most similar
  """
  search = Similarity()
  wiki_search = wikipedia.search(query, results=1, suggestion=True)
  pattern = r"'(.*?)'"
  wiki_query = re.findall(pattern, str(wiki_search))
  wiki_page = wikipedia.page(str(wiki_query))
  text_without_equals = str(wiki_page.content).replace('=', '')
  wiki_text = re.split(r'(?<=[.!?]) +', text_without_equals)
  similar_text = ""
  for sim in search.infer([user], wiki_text, top=top):
      similar_text += sim
  return similar_text

  
