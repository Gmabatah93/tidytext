library(tidyverse)
library(lubridate)
library(purrr)
theme_set(theme_minimal())
library(tidytext)
library(wordcloud)
library(topicmodels)

#
# Tidytext Format ----

# Data
orginal_books <- janeaustenr::austen_books() %>% 
  group_by(book) %>% 
  mutate(line_num = row_number(),
         chapter = cumsum(str_detect(text,
                                     regex("^chapter [\\divxlc]",
                                     ignore_case = TRUE)))) %>% 
  ungroup()
hgwells <- gutenbergr::gutenberg_download(c(35, 36, 5230, 159))
bronte <- gutenbergr::gutenberg_download(c(1260, 768, 969, 9182, 767))

# Tokenization
# - Jane Austen
tidy_books <- orginal_books %>% 
  unnest_tokens(word, text) %>%  
  anti_join(stop_words)
tidy_books %>% count(word, sort = TRUE)
tidy_books %>% 
  count(word, sort = TRUE) %>% 
  filter(n > 600) %>% 
  mutate(word = reorder(word,n)) %>% 
  ggplot(aes(n, word)) +
  geom_col() +
  labs(y = NULL)

# - HG Wells
tidy_hgwells <- hgwells %>%  
  unnest_tokens(word, text) %>% 
  anti_join(stop_words)
tidy_hgwells %>% count(word, sort = TRUE)
tidy_hgwells %>% 
  count(word, sort = TRUE) %>% 
  filter(n > 150) %>% 
  mutate(word = reorder(word,n)) %>% 
  ggplot(aes(n, word)) +
  geom_col() +
  labs(y = NULL)

# - Bronte Sisters
tidy_bronte <- bronte %>%  
  unnest_tokens(word, text) %>% 
  anti_join(stop_words)
tidy_bronte %>% count(word, sort = TRUE)
tidy_bronte %>% 
  count(word, sort = TRUE) %>% 
  filter(n > 400) %>% 
  mutate(word = reorder(word,n)) %>% 
  ggplot(aes(n, word)) +
  geom_col() +
  labs(y = NULL)

# TF-IDF ----

# Data
books <- janeaustenr::austen_books()
# Tokenization
book_words <-  books %>% 
  unnest_tokens(word, text) %>%
  count(book, word, sort = TRUE)
total_words <- book_words %>% 
  group_by(book) %>% 
  summarize(total = sum(n))
book_words <- left_join(book_words, total_words)

book_words %>% 
  ggplot(aes(n/total, fill = book)) +
  geom_histogram(show.legend = FALSE) +
  xlim(NA, 0.0009) +
  facet_wrap(~book, ncol = 2, scales = "free_y")

# tf-idf
book_tf_idf <- book_words %>%
  bind_tf_idf(word, book, n)
book_tf_idf %>%
  select(-total) %>%
  arrange(desc(tf_idf))
# - visual
library(forcats)
book_tf_idf %>%
  group_by(book) %>%
  slice_max(tf_idf, n = 15) %>%
  ungroup() %>%
  ggplot(aes(tf_idf, fct_reorder(word, tf_idf), fill = book)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~book, ncol = 2, scales = "free") +
  labs(x = "tf-idf", y = NULL)




# Data
physics <- gutenbergr::gutenberg_download(c(37729, 14725, 13476, 30155), 
                                          meta_fields = "author")
# Tokenization
physics_words <- physics %>% 
  unnest_tokens(word, text) %>% 
  count(author, word, sort = TRUE)
# TF-IDF
physics_tf_idf <- physics_words %>%
  bind_tf_idf(word, author, n) 
# - visual
physics_tf_idf %>% 
  mutate(author = factor(author, levels = c("Galilei, Galileo",
                                            "Huygens, Christiaan", 
                                            "Tesla, Nikola",
                                            "Einstein, Albert"))) %>% 
  group_by(author) %>% 
  slice_max(tf_idf, n = 15) %>% 
  ungroup() %>%
  mutate(word = reorder(word, tf_idf)) %>%
  ggplot(aes(tf_idf, word, fill = author)) +
  geom_col(show.legend = FALSE) +
  labs(x = "tf-idf", y = NULL) +
  facet_wrap(~author, ncol = 2, scales = "free")

#
# n-grams & correlations ----

# Data
austen <- janeaustenr::austen_books() 

# Toeknization
austen_bigrams <- austen %>% 
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>% 
  separate(bigram, c("word1","word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>% 
  filter(!word2 %in% stop_words$word)

austen_bigrams_united <- austen_bigrams %>% 
  unite(bigram, word1, word2, sep = " ")

austen_bigrams_tfidf <- austen_bigrams_united %>% 
  count(book, bigram) %>% 
  bind_tf_idf(bigram, book, n) %>% 
  arrange(-tf_idf)

library(forcats)
austen_bigrams_tfidf %>%
  group_by(book) %>%
  slice_max(tf_idf, n = 15) %>%
  ungroup() %>%
  ggplot(aes(tf_idf, fct_reorder(bigram, tf_idf), fill = book)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~book, ncol = 2, scales = "free") +
  labs(x = "tf-idf", y = NULL)

# EDA
austen_bigrams %>% 
  filter(word2 == "street") %>% 
  count(book, word1, sort = TRUE)

austen %>% 
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>% 
  separate(bigram, c("word1","word2"), sep = " ") %>%
  filter(word1 == "not") %>%
  count(word1, word2, sort = TRUE)

# Sentiment Analysis
AFINN <- get_sentiments("afinn")
# - not words
not_words <- austen %>% 
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>% 
  separate(bigram, c("word1","word2"), sep = " ") %>%
  filter(word1 == "not") %>%
  inner_join(AFINN, by = c(word2 = "word")) %>%
  count(word2, value, sort = TRUE)
not_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(20) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(n * value, word2, fill = n * value > 0)) +
  geom_col(show.legend = FALSE) +
  labs(x = "Sentiment value * number of occurrences",
       y = "Words preceded by \"not\"")
# - negation words
negation_words <- c("not", "no", "never", "without")

negated_words <- austen %>% 
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>% 
  separate(bigram, c("word1","word2"), sep = " ") %>%
  filter(word1 %in% negation_words) %>%
  inner_join(AFINN, by = c(word2 = "word")) %>%
  count(word1, word2, value, sort = TRUE)
negated_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(50) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(n * value, word2, fill = n * value > 0)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~word1) +
  labs(x = "Sentiment value * number of occurrences",
       y = "Words preceded by \"not\"")

library(igraph)
library(ggraph)
bigram_graph <- bigrams_count %>%
  filter(n > 20) %>%
  graph_from_data_frame()

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

set.seed(2020)
a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()


# Converting to and from non-tidy formats ----
library(tm)

# Data
data("AssociatedPress", package = "topicmodels")
AssociatedPress
terms <- Terms(AssociatedPress)
head(terms)

data("data_corpus_inaugural", package = "quanteda")
inaug_dfm <- quanteda::dfm(data_corpus_inaugural, verbose = FALSE)

data("acq", package = "tm")
acq[[1]]


# Tidying DocumentTermMatrix objects "only non-zero values are included"
ap_tidy <- tidy(AssociatedPress) 
# - Sentiment Analysis
ap_sentiments <- ap_tidy %>%
  inner_join(get_sentiments("bing"), by = c(term = "word"))
ap_sentiments %>%
  count(sentiment, term, wt = count) %>%
  ungroup() %>%
  filter(n >= 200) %>%
  mutate(n = ifelse(sentiment == "negative", -n, n)) %>%
  mutate(term = reorder(term, n)) %>%
  ggplot(aes(n, term, fill = sentiment)) +
  geom_col() +
  labs(x = "Contribution to sentiment", y = NULL)



# Tidying Document-Feature Matrix objects
inuag_tidy <- tidy(inaug_dfm)
# - tokenization
inuag_tidy %>% 
  bind_tf_idf(term, document, count) %>% 
  arrange(-tf_idf) %>% 
  group_by(document) %>% 
  top_n(n = 15,wt = tf_idf) %>% 
  ungroup() %>%
  filter(document %in% c("1861-Lincoln","1933-Roosevelt","1961-Kennedy","2009-Obama")) %>% 
  ggplot(aes(tf_idf, forcats::fct_reorder(term, tf_idf), fill = document)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~document,scales = "free") +
  labs(x = NULL, y = NULL) +
  theme_bw()
# - Total number of words within each year & how word frequency changes over the years
words <- c("god", "america", "foreign", "union", "constitution", "freedom")
year_term_counts <- inuag_tidy %>% 
  extract(document, "year", "(\\d+)", convert = TRUE) %>% 
  complete(year, term, fill = list(count = 0)) %>% 
  group_by(year) %>% 
  mutate(year_total = sum(count))
year_term_counts %>% 
  filter(term %in% words) %>% 
  ggplot(aes(year, count / year_total)) +
  geom_point() +
  geom_smooth() +
  facet_wrap(~term, scales = "free_y") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(y = "% frequency of word in inaugural address") +
  theme_bw()



# Casting tidy text data into a Matrix
ap_tidy %>% cast_dtm(document = document, 
                     term = term, 
                     value = count)
ap_tidy %>% cast_dfm(document = document,
                     term = term,
                     value = count)
# - sparsity
library(Matrix)
ap_tidy %>% cast_sparse(row = document,
                        column = term, 
                        value = count) %>% 
  dim()




# Tidying corpus objects with metadata
# - tidy
acq_tidy <- tidy(acq)
# - tokenization
acq_tokens <- acq_tidy %>%
  select(-places) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word")
# - word frequency
acq_tokens %>% count(word, sort = TRUE)
# - tf_idf
acq_tf_idf <- acq_tokens %>% 
  count(id, word) %>% 
  bind_tf_idf(term = word,
              document =  id,
              n = n) %>% 
  arrange(desc(tf_idf))

ids <- c("110","372","362","496","45","302","331","448","10","393")
acq_tf_idf %>%
  group_by(id) %>% 
  top_n(n = 15, wt = tf_idf) %>% 
  ungroup() %>% 
  filter(id %in% ids) %>% 
  ggplot(aes(tf_idf, forcats::fct_reorder(word, tf_idf), fill = id)) +
  geom_col(show.legend = F) +
  facet_wrap(~id, nrow = 2, scales = "free") +
  labs(y = NULL, x = NULL)








# EXAMPLE: Jane Austenr
# - data
austen <- janeaustenr::austen_books()
# - tokenize
austen_tidy <- austen %>% 
  unnest_tokens(word, text) %>%
  count(book, word)
# - cast DTM
austen_tidy %>% 
  cast_dtm(document = book, 
           term = word, 
           value = n)


# EXAMPLE: Mnining Financial Articles
library(tm.plugin.webmining)
library(purrr)

company <- c("Microsoft", "Apple", "Google", "Amazon", "Facebook",
             "Twitter", "IBM", "Yahoo", "Netflix")
symbol  <- c("MSFT", "AAPL", "GOOG", "AMZN", "FB", 
             "TWTR", "IBM", "YHOO", "NFLX")

download_articles <- function(symbol) {
  WebCorpus(GoogleFinanceSource(paste0("NASDAQ:", symbol)))
}

stock_articles <- tibble(company = company,
                         symbol = symbol) %>%
  mutate(corpus = map(symbol, download_articles))

# Sentiment Analysis ----

# Lexicons
get_sentiments("afinn")
get_sentiments("bing")
get_sentiments("nrc")

# Data
books <- janeaustenr::austen_books() %>% 
  group_by(book) %>% 
  mutate(
    linenumber = row_number(),
    chapter = cumsum(str_detect(text, 
                                regex("^chapter [\\divxlc]", 
                                ignore_case = TRUE))))

# Tokenization
books_tidy <- books %>% 
  ungroup() %>%
  unnest_tokens(word, text)

# Senitment: nrc
nrc_joy <- 
  get_sentiments("nrc") %>% 
  filter(sentiment == "joy")

books_tidy %>% 
  filter(book == "Emma") %>% 
  inner_join(nrc_joy) %>% 
  count(word, sort = TRUE)

# Senitment: bing
books_bing <-  books_tidy %>% 
  inner_join(get_sentiments("bing")) %>% 
  count(book, index = linenumber %/% 80, sentiment) %>% 
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>% 
  mutate(sentiment = positive - negative)

books_bing %>% 
  ggplot(aes(index, sentiment, fill = book)) +
  geom_col(show.legend = F) +
  facet_wrap(~book, ncol = 2, scales = "free_x") +
  ggtitle("how the plot of each novel changes toward more positive or negative sentiment over the trajectory of the story.")

# EDA: How does sentiment changes across the narrative arc of Pride and Prejudice
pride_prejudice_tidy <- books_tidy %>% 
  filter(book == "Pride & Prejudice")
# - afinn
pride_prejudice_afinn <- pride_prejudice_tidy %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(index = linenumber %/% 80) %>% 
  summarise(sentiment = sum(value)) %>% 
  mutate(method = "AFINN")
# - bing & nrc
pride_prejudice_bing_and_nrc <- 
  bind_rows(pride_prejudice_tidy %>% 
              inner_join(get_sentiments("bing")) %>% 
              mutate(method = "Bing et al."),
            pride_prejudice_tidy %>% 
              inner_join(get_sentiments("nrc") %>% filter(sentiment %in% c("positive", "negative"))) %>% 
              mutate(method = "NRC")) %>% 
  count(method, index = linenumber %/% 80, sentiment) %>% 
  pivot_wider(names_from = sentiment,
              values_from = n,
              values_fill = 0) %>% 
  mutate(sentiment = positive - negative)

# - visual
bind_rows(pride_prejudice_afinn, pride_prejudice_bing_and_nrc) %>% 
  ggplot(aes(index, sentiment, fill = method)) +
  geom_col(show.legend = F) +
  facet_wrap(~method, ncol = 1, scales = "free_y")

# EDA: Most common positive and negative words
books_bing_word_counts <- books_tidy %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

books_bing_word_counts %>% 
  group_by(sentiment) %>% 
  slice_max(n, n = 10) %>% 
  ungroup() %>% 
  ggplot(aes(n, forcats::fct_reorder(word, n), fill = sentiment)) +
  geom_col(show.legend = F) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = NULL, x = NULL) +
  theme_bw()

custom_stop_words <- bind_rows(tibble(word = c("miss"),  
                                      lexicon = c("custom")), 
                               stop_words)

# WordCloud
tidy_books %>% 
  anti_join(stop_words) %>% 
  count(word) %>% 
  with(wordcloud(word, n, max.words = 100))

library(reshape2)
tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("gray20", "gray80"),
                   max.words = 100)

# Topic Models ----

# Data
data("AssociatedPress")
AssociatedPress # collection of 2246 news articles from an American news agency, mostly published around 1988

# LDA
ap_lda <- LDA(AssociatedPress, k = 2, control = list(seed = 1234))

# - Topic-word: (1 = Business/Financial | 2 = Political)
ap_topics <- tidy(ap_lda, matrix = "beta")
ap_topics_terms <- ap_topics %>% 
  group_by(topic) %>% 
  slice_max(beta, n = 10) %>% 
  ungroup() %>% 
  arrange(topic, -beta)

ap_topics_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

# - Document-topic 
ap_documents <- tidy(ap_lda, matrix = "gamma")
tidy(AssociatedPress) %>%
  filter(document == 6) %>%
  arrange(desc(count))

# EXAMPLE: The Great Library Heist ----

# Data
library(gutenbergr)
titles <- c("Twenty Thousand Leagues under the Sea", 
            "The War of the Worlds",
            "Pride and Prejudice", 
            "Great Expectations")
books <- gutenberg_works(title %in% titles) %>% 
  gutenberg_download(meta_fields = "title")
books %>% count(title)

# Text Preprocessing
# - Each document represents 1 chapter
by_chapter <- books %>%
  group_by(title) %>%
  mutate(chapter = cumsum(str_detect(
    text, regex("^chapter ", ignore_case = TRUE)
  ))) %>%
  ungroup() %>%
  filter(chapter > 0) %>%
  unite(document, title, chapter)
# - Bag of Words
by_chapter_word <- by_chapter %>%
  unnest_tokens(word, text)

by_chapter_word_counts <- by_chapter_word %>%
  anti_join(stop_words) %>%
  count(document, word, sort = TRUE) %>%
  ungroup()


# LDA
# - Document Term Matrix
chapters_dtm <- by_chapter_word_counts %>%
  cast_dtm(document, word, n)
# - 4 Books -> 4 Chapters
chapters_lda <- LDA(chapters_dtm, k = 4, control = list(seed = 1234))
# - Topic-word
chapter_topics <- tidy(chapters_lda, matrix = "beta")
chapter_topics_terms <- chapter_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 5) %>% 
  ungroup() %>%
  arrange(topic, -beta)

chapter_topics_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(title = "Top ")

# - Document-topic 
chapters_gamma <- tidy(chapters_lda, matrix = "gamma")
chapters_gamma <- chapters_gamma %>%
  separate(document, c("title", "chapter"), sep = "_", convert = TRUE)
chapters_gamma %>%
  mutate(title = reorder(title, gamma * topic)) %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_boxplot() +
  facet_wrap(~ title) +
  labs(x = "topic", y = expression(gamma))

# EXAMPLE: Twitter ----

# Data
tweets_julia <- read_csv(file = "https://raw.githubusercontent.com/dgrtwo/tidy-text-mining/master/data/tweets_julia.csv")
tweets_dave <- read_csv("https://raw.githubusercontent.com/dgrtwo/tidy-text-mining/master/data/tweets_dave.csv")
tweets <- bind_rows(tweets_julia %>% mutate(person = "Julia"),
                    tweets_dave %>% mutate(person = "David")) %>% 
            mutate(timestamp = ymd_hms(timestamp))

# EDA
tweets %>% 
  ggplot(aes(timestamp, fill = person)) +
  geom_histogram(position = "identity", bins = 20, show.legend = FALSE) +
  facet_wrap(~person, ncol = 1) +
  ggtitle(label = "Frequency of Tweets")

# Text Preprocessing 
remove_reg <- "&amp;|&lt;|&gt;"
tidy_tweets <- tweets %>% 
  # remove retweets
  filter(!str_detect(text, "^RT")) %>%
  # remove amps etc...
  mutate(text = str_remove_all(text, remove_reg)) %>%
  # bag of words (retain hashtags and mentions of usernames with the @ symbol)
  unnest_tokens(word, text, token = "tweets") %>%
  # remove stop words 
  filter(!word %in% stop_words$word,
         !word %in% str_remove_all(stop_words$word, "'"),
         str_detect(word, "[a-z]"))



# Word Frequencies for each person
frequency <- tidy_tweets %>% 
  group_by(person) %>% 
  count(word, sort = TRUE) %>% 
  left_join(tidy_tweets %>% 
              group_by(person) %>% 
              summarise(total = n())) %>% 
  mutate(freq = n/total)

frequency %>% 
  select(person, word, freq) %>% 
  pivot_wider(names_from = person, values_from = freq) %>% 
  arrange(Julia, David) %>% 
  ggplot(aes(Julia, David)) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.25, height = 0.25) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = scales::percent_format()) +
  scale_y_log10(labels = scales::percent_format()) + 
  geom_abline(color = "red")




# Tweets (2016)
tidy_tweets_2016 <- tidy_tweets %>% 
  filter(timestamp >= as.Date("2016-01-01"),
         timestamp < as.Date("2017-01-01"))

word_ratio <- tidy_tweets_2016 %>% 
  # remove twitter usernames
  filter(!str_detect(word, "^@")) %>% 
  count(word, person) %>% 
  group_by(word) %>% 
  filter(sum(n) >= 10) %>% 
  ungroup() %>% 
  pivot_wider(names_from = person, values_from = n, values_fill = 0) %>% 
  # calc log ratio
  mutate_if(is.numeric, list(~(. + 1) / (sum(.) + 1))) %>% 
  mutate(logratio = log(David / Julia)) %>% 
  arrange(desc(logratio))

# - What are some words that have been about equally likely to come from David or Julia’s account during 2016 ?
word_ratio %>% 
  arrange(abs(logratio))
# - Which words are most likely to be from Julia’s account or from David’s account ?
word_ratio %>%
  group_by(logratio < 0) %>%
  slice_max(abs(logratio), n = 15) %>% 
  ungroup() %>%
  mutate(word = reorder(word, logratio)) %>%
  ggplot(aes(word, logratio, fill = logratio < 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  ylab("log odds ratio (David/Julia)") +
  scale_fill_discrete(name = "", labels = c("David", "Julia"))




# Changes in word use
words_by_time <- tidy_tweets_2016 %>% 
  filter(!str_detect(word, "^@")) %>%
  # define a new time variable in the data frame that defines which unit of time each tweet was posted in.
  mutate(time_floor = floor_date(timestamp, unit = "1 month")) %>%
  count(time_floor, person, word) %>%
  group_by(person, time_floor) %>%
  # total number of words used in each time bin
  mutate(time_total = sum(n)) %>%
  group_by(person, word) %>%
  # total number of times each word was used
  mutate(word_total = sum(n)) %>%
  ungroup() %>%
  rename(count = n) %>%
  filter(word_total > 30)

# - MODEL: Was a given word mentioned in a given time bin ? Yes or no ? How does the count of word mentions depend on time ?”
nested_data <- words_by_time %>%
  nest(-word, -person)

nested_models <- nested_data %>%
  mutate(models = map(data, ~ glm(cbind(count, time_total) ~ time_floor, ., 
                                  family = "binomial")))
# - pull out the slopes for each of these models and find the important ones
slopes <- nested_models %>%
  mutate(models = map(models, tidy)) %>%
  unnest(cols = c(models)) %>%
  filter(term == "time_floor") %>%
  mutate(adjusted.p.value = p.adjust(p.value))
top_slopes <- slopes %>% 
  filter(adjusted.p.value < 0.05)

words_by_time %>%
  inner_join(top_slopes, by = c("word", "person")) %>%
  filter(person == "David") %>%
  ggplot(aes(time_floor, count/time_total, color = word)) +
  geom_line(size = 1.3) +
  labs(x = NULL, y = "Word frequency")
words_by_time %>%
  inner_join(top_slopes, by = c("word", "person")) %>%
  filter(person == "Julia") %>%
  ggplot(aes(time_floor, count/time_total, color = word)) +
  geom_line(size = 1.3) +
  labs(x = NULL, y = "Word frequency")




# EXAMPLE: Favorites and Retweets ----

# Data
tweets_julia <- read_csv("https://raw.githubusercontent.com/dgrtwo/tidy-text-mining/master/data/juliasilge_tweets.csv")
tweets_david <- read_csv("https://raw.githubusercontent.com/dgrtwo/tidy-text-mining/master/data/drob_tweets.csv")
tweets <- bind_rows(tweets_julia %>% mutate(person = "Julia"),
                    tweets_dave %>% mutate(person = "David")) %>%
  mutate(created_at = ymd_hms(created_at))

# Text Preprocessing 
tidy_tweets <- tweets %>% 
  # remove all retweets and replies
  filter(!str_detect(text, "^(RT|@)")) %>%
  mutate(text = str_remove_all(text, remove_reg)) %>%
  unnest_tokens(word, text, token = "tweets", strip_url = TRUE) %>%
  filter(!word %in% stop_words$word,
         !word %in% str_remove_all(stop_words$word, "'"))

# EDA
# - Total Number of Retweets
totals <- tidy_tweets %>% 
  group_by(person, id) %>% 
  summarise(rts = first(retweets)) %>% 
  group_by(person) %>% 
  summarise(total_rts = sum(rts))

# EXAMPLE: NASA ----

# Data
library(jsonlite)
metadata <- fromJSON("https://data.nasa.gov/data.json")

nasa_title <- tibble(title = metadata$dataset$title) %>% 
  mutate(id = seq(1, nrow(.))) %>% 
  select(id, title)
nasa_desc <- tibble(desc = metadata$dataset$description) %>% 
  mutate(id = seq(1, nrow(.))) %>% 
  select(id, desc)
nasa_keyword <- tibble(keyword = metadata$dataset$keyword) %>% unnest(keyword)

# Tokenization
my_stopwords <- tibble(word = c(as.character(1:10), 
                                "v1", "v03", "l2", "l3", "l4", "v5.2.0", 
                                "v003", "v004", "v005", "v006", "v7",
                                stop_words$word))
nasa_title_tokens <- nasa_title %>% 
  unnest_tokens(output = word, input = title) %>% 
  anti_join(my_stopwords)
nasa_desc_tokens <- nasa_desc %>% 
  unnest_tokens(word, desc) %>% 
  anti_join(my_stopwords)

# EDA 
# - counts
nasa_title_tokens %>% count(word, sort = TRUE)
nasa_desc_tokens %>% count(word, sort = TRUE)
nasa_keyword %>% group_by(keyword) %>% count(sort = TRUE)
# - word co-ocurrences and correlations
library(widyr)
library(igraph)
library(ggraph)

title_words_pairs <- nasa_title_tokens %>% 
  pairwise_count(word, id, sort = TRUE, upper = FALSE)
desc_word_pairs <- nasa_desc_tokens %>% 
  pairwise_count(word, id, sort = TRUE, upper = FALSE)

title_words_pairs %>% 
  filter(n >= 250) %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(edge_alpha = n, edge_width = n), edge_colour = "cyan4") +
  geom_node_point(size = 5) +
  geom_node_text(aes(label = name), repel = TRUE, 
                 point.padding = unit(0.2, "lines")) +
  theme_void()

set.seed(1234)
desc_word_pairs %>%
  filter(n >= 5000) %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(edge_alpha = n, edge_width = n), edge_colour = "darkred") +
  geom_node_point(size = 5) +
  geom_node_text(aes(label = name), repel = TRUE,
                 point.padding = unit(0.2, "lines")) +
  theme_void()

keyword_pairs <- nasa_keyword %>% 
  pairwise_count(keyword, id, sort = TRUE, upper = FALSE)

# EXAMPLE: Usenet Bulletin Boards ----
