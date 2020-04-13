library(tidyverse)
library(tidytext)
library(ngram)
library(stopwords)
library(tm)
library(wordcloud)

# This is a script used to anayze the caption vocabulary

captions_df <- read_delim('Flickr8k.token.txt', delim = '\t', col_names = FALSE)
names(captions_df) <- c('Image', 'Caption')

captions_df <- captions_df %>%
  mutate(Image = str_remove(Image, '.{2}$'),
         Caption = str_remove(Caption, '.{2}$'))

train_df <- read_delim('Flickr_8k.trainimages.txt', delim = '\t', col_names = FALSE)
names(train_df) <- c('Train_Image')

test_df <- read_delim('Flickr_8k.testimages.txt', delim = '\t', col_names = FALSE)
names(test_df) <- c('Test_Image')

captions_df <- captions_df %>%
  mutate(Train_Test = if_else(captions_df$Image %in% (train_df %>% pull()), 'Train', 'Test'))

# Number of images
nrow(captions_df) / 5

# Freq distribution of captions per image
captions_df %>% 
  group_by(Image) %>% 
  mutate(n_captions = 1) %>% 
  group_by(Image) %>% 
  summarise(N_Captions = sum(n_captions)) %>% 
  pull(N_Captions) %>% 
  unique()

# Distribution length of stopwords

word_count <- function(text) {
  length(unlist(str_split(text, '\\s')))
}

word_count <- Vectorize(word_count)

# Length of the captions
captions_df %>% 
  mutate(Length = word_count(Caption)) %>% 
  arrange(-Length) %>% 
  mutate(Id = row_number()) %>% 
  ggplot(aes(x = Id, y = Length, col = Train_Test)) + 
    geom_line(lwd = 1) + 
  facet_wrap(.~Train_Test) + 
  theme_bw() + 
  labs(title = "Distributions of caption lengths", x = '', y = 'Words')

# Length of the captions
captions_df %>% 
  mutate(Length = word_count(Caption)) %>% 
  ggplot(aes(x = Length, col = Train_Test)) + 
  geom_histogram(bins = 35) + 
  facet_wrap(.~Train_Test) + 
  theme_bw() + 
  labs(title = "Caption lengths distributions", x = 'Number of words per caption', y = 'Frequency', col = 'Datasets')


captions_df %>%
  filter(Train_Test == 'Train') %>% 
  dplyr::select(Caption) %>%
  unnest_tokens(word, Caption) %>% 
  anti_join(stop_words, by = c("word" = "word")) %>% 
  count(word, sort = TRUE) %>%
  top_n(25) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) +
  geom_col(col = 'Tomato') +
  xlab(NULL) +
  coord_flip() +
  labs(x = "Count",
       y = "Unique words",
       title = "Top 25 most frequent words in train dataset [7.489 words] [Stopwords excluded]") + 
  labs(x = '', y = 'Frequency') + 
  theme_bw() + 
  theme(legend.position = "none")

captions_df %>%
  filter(Train_Test == 'Test') %>% 
  dplyr::select(Caption) %>%
  unnest_tokens(word, Caption) %>% 
  anti_join(stop_words, by = c("word" = "word")) %>% 
  count(word, sort = TRUE) %>%
  top_n(25) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) + 
  geom_col(col = '#00BFC4') + 
  xlab(NULL) + 
  coord_flip() + 
  labs(x = "Count",
       y = "Unique words",
       title = "Top 25 most frequent words in test dataset [4.727] [Stopwords excluded]") + 
  labs(x = '', y = 'Frequency') + 
  theme_bw() + 
  theme(legend.position = "none")

train_words <- captions_df %>%
  filter(Train_Test == 'Train') %>% 
  dplyr::select(Caption) %>%
  unnest_tokens(word, Caption) %>% 
  anti_join(stop_words, by = c("word" = "word")) %>% 
  count(word, sort = TRUE)

test_words <- captions_df %>% 
  filter(Train_Test == 'Test') %>% 
  dplyr::select(Caption) %>% 
  unnest_tokens(word, Caption) %>% 
  anti_join(stop_words, by = c("word" = "word")) %>% 
  count(word, sort = TRUE)

test_words %>% 
  filter(!test_words$word %in% train_words$word) %>% 
  top_n(25, n) %>% 
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) + 
  geom_col(col = '#fcba03') + 
  xlab(NULL) + 
  coord_flip() + 
  labs(x = "Count",
       y = "Unique words",
       title = "Top 25 most frequent unique words in test set [1.194, 25.26%]") + 
  labs(x = '', y = 'Frequency') + 
  theme_bw() + 
  theme(legend.position = "none")

# plot the top 15 words
captions_df %>%
  count(word, sort = TRUE) %>%
  top_n(15) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip() +
  labs(x = "Count",
       y = "Unique words",
       title = "Count of unique words found in tweets")

sum(!(test_words$word %in% train_words$word)) / length(test_words$word) * 100