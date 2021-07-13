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
