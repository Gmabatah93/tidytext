library(tidytext)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
theme_set(theme_minimal())
library(wordcloud)

#
# Tidytext Format ----
text <- c("Because I could not stop for Death -",
          "He kindly stopped for me -",
          "The Carriage held but just Ourselves -",
          "and Immortality")
text_df <- tibble(line = 1:4, text = text)
text_df %>% 
  unnest_tokens(word, text)

orginal_books <- janeaustenr::austen_books() %>% 
  group_by(book) %>% 
  mutate(line_num = row_number(),
         chapter = cumsum(str_detect(text,
                                     regex("^chapter [\\divxlc]",
                                     ignore_case = TRUE)))) %>% 
  ungroup()
tidy_books <- orginal_books %>% 
  unnest_tokens(word, text)
# - remove stop words
tidy_books <- tidy_books %>% 
  anti_join(stop_words)
tidy_books %>% 
  count(word, sort = TRUE)
# - visual
tidy_books %>% 
  count(word, sort = TRUE) %>% 
  filter(n > 600) %>% 
  mutate(word = reorder(word,n)) %>% 
  ggplot(aes(n, word)) +
  geom_col() +
  labs(y = NULL)


hgwells <- gutenbergr::gutenberg_download(c(35, 36, 5230, 159))
tidy_hgwells <- hgwells %>%  
  unnest_tokens(word, text) %>% 
  anti_join(stop_words)
tidy_hgwells %>% 
  count(word, sort = TRUE)

bronte <- gutenbergr::gutenberg_download(c(1260, 768, 969, 9182, 767))
tidy_bronte <- bronte %>%  
  unnest_tokens(word, text) %>% 
  anti_join(stop_words)
tidy_bronte %>% 
  count(word, sort = TRUE)

frequency <- bind_rows(mutate(tidy_bronte, author = "Brontë Sisters"),
                       mutate(tidy_hgwells, author = "H.G. Wells"), 
                       mutate(tidy_books, author = "Jane Austen")) %>% 
  mutate(word = str_extract(word, "[a-z']+")) %>%
  count(author, word) %>%
  group_by(author) %>%
  mutate(proportion = n / sum(n)) %>% 
  select(-n) %>% 
  pivot_wider(names_from = author, values_from = proportion) %>%
  pivot_longer(`Brontë Sisters`:`H.G. Wells`,
               names_to = "author", values_to = "proportion")

library(scales)
ggplot(frequency, aes(x = proportion, y = `Jane Austen`, 
                      color = abs(`Jane Austen` - proportion))) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  scale_color_gradient(limits = c(0, 0.001), 
                       low = "darkslategray4", high = "gray75") +
  facet_wrap(~author, ncol = 2) +
  theme(legend.position="none") +
  labs(y = "Jane Austen", x = NULL)
# - Correlation
cor.test(data = frequency[frequency$author == "Brontë Sisters",],
         ~ proportion + `Jane Austen`)
cor.test(data = frequency[frequency$author == "H.G. Wells",], 
         ~ proportion + `Jane Austen`)

# Sentiment Analysis ----
get_sentiments("afinn")
get_sentiments("bing")
get_sentiments("nrc")

tidy_books <- janeaustenr::austen_books() %>% 
  group_by(book) %>% 
  mutate(
    linenumber = row_number(),
    chapter = cumsum(str_detect(text, 
                                regex("^chapter [\\divxlc]", 
                                ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text)

# nrc
nrc_joy <- get_sentiments("nrc") %>% 
  filter(sentiment == "joy")

tidy_books %>% 
  filter(book == "Emma") %>% 
  inner_join(nrc_joy) %>% 
  count(word, sort = TRUE)

# bing
jane_austen_sentiment <- tidy_books %>% 
  inner_join(get_sentiments("bing")) %>% 
  count(book, index = linenumber %/% 80, sentiment) %>% 
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>% 
  mutate(sentiment = positive - negative)

jane_austen_sentiment %>% 
  ggplot(aes(index, sentiment, fill = book)) +
  geom_col(show.legend = F) +
  facet_wrap(~book, ncol = 2, scales = "free_x")

# 3 sentiments
pride_prejudice <- tidy_books %>% 
  filter(book == "Pride & Prejudice")

afinn <- pride_prejudice %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(index = linenumber %/% 80) %>% 
  summarise(sentiment = sum(value)) %>% 
  mutate(method = "AFINN")

bing_and_nrc <- bind_rows(
  pride_prejudice %>% 
    inner_join(get_sentiments("bing")) %>% 
    mutate(method = "Bing et al."),
  pride_prejudice %>% 
    inner_join(get_sentiments("nrc") %>% 
                 filter(sentiment %in% c("positive", "negative"))
               ) %>% 
    mutate(method = "NRC")) %>% 
  count(method, index = linenumber %/% 80, sentiment) %>% 
  pivot_wider(names_from = sentiment,
              values_from = n,
              values_fill = 0) %>% 
  mutate(sentiment = positive - negative)

# - Visual
bind_rows(afinn, bing_and_nrc) %>% 
  ggplot(aes(index, sentiment, fill = method)) +
  geom_col(show.legend = F) +
  facet_wrap(~method, ncol = 1, scales = "free_y")

# Most common positive and negative words
bing_word_counts <- tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

bing_word_counts %>% 
  group_by(sentiment) %>% 
  slice_max(n, n = 10) %>% 
  ungroup() %>% 
  ggplot(aes(n, word, fill = sentiment)) +
  geom_col(show.legend = F) +
  facet_wrap(~sentiment, scales = "free_y")

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

# TF-IDF ----
book_words <- janeaustenr::austen_books() %>%
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

# Zipf’s law
freq_by_rank <- book_words %>% 
  group_by(book) %>% 
  mutate(rank = row_number(), 
         `term frequency` = n/total) %>%
  ungroup()
freq_by_rank %>% 
  ggplot(aes(rank, `term frequency`, color = book)) + 
  geom_abline(intercept = -0.62, slope = -1.1, 
              color = "gray50", linetype = 2) +
  geom_line(size = 1.1, alpha = 0.8, show.legend = FALSE) + 
  scale_x_log10() +
  scale_y_log10()

# tf-idf
book_tf_idf <- book_words %>%
  bind_tf_idf(word, book, n)
book_tf_idf %>%
  select(-total) %>%
  arrange(desc(tf_idf))

# - Visual
library(forcats)

book_tf_idf %>%
  group_by(book) %>%
  slice_max(tf_idf, n = 15) %>%
  ungroup() %>%
  ggplot(aes(tf_idf, fct_reorder(word, tf_idf), fill = book)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~book, ncol = 2, scales = "free") +
  labs(x = "tf-idf", y = NULL)

# n-grams & correlations ----
austen_bigrams <- janeaustenr::austen_books() %>% 
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

austen_bigrams %>% 
  count(bigram, sort = TRUE)

bigrams_seperated <- austen_bigrams %>% 
  separate(bigram, c("word1","word2"), sep = " ")

bigrams_filtered <- bigrams_seperated %>% 
  filter(!word1 %in% stop_words$word) %>% 
  filter(!word2 %in% stop_words$word)

bigrams_count <- bigrams_filtered %>% 
  count(word1, word2, sort = TRUE)

bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

janeaustenr::austen_books() %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
  count(word1, word2, word3, sort = TRUE)

bigrams_filtered %>%
  filter(word2 == "street") %>%
  count(book, word1, sort = TRUE)

bigram_tf_idf <- bigrams_united %>%
  count(book, bigram) %>%
  bind_tf_idf(bigram, book, n) %>%
  arrange(desc(tf_idf))

bigrams_seperated %>%
  filter(word1 == "not") %>%
  count(word1, word2, sort = TRUE)

AFINN <- get_sentiments("afinn")
not_words <- bigrams_seperated %>%
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

negation_words <- c("not", "no", "never", "without")

negated_words <- bigrams_seperated %>%
  filter(word1 %in% negation_words) %>%
  inner_join(AFINN, by = c(word2 = "word")) %>%
  count(word1, word2, value, sort = TRUE)

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
data("AssociatedPress", package = "topicmodels")
AssociatedPress
terms <- Terms(AssociatedPress)
head(terms)

ap_td <- tidy(AssociatedPress) 

ap_sentiments <- ap_td %>%
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

ap_td %>%
  cast_dtm(document, term, count)
