# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import networkx as nx

import text_preprocessing.preprocess as text_process


class GraphUtils(object):
    def draw_graph(self, graph, plot_file=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.axis('off')
        pos = nx.spring_layout(graph, k=0.2, iterations=20)
        nx.draw_networkx(graph, pos=pos, arrows=False, with_labels=True, node_size=15, alpha=0.65, width=0.2,
                         edge_color='b', font_size=10)
        if plot_file is not None:
            plt.savefig(plot_file)
        plt.show()

    def sort_by_value(self, item_list, order='desc'):
        """
        A utility function to sort lists by their value.
        Args:
            item_list:
            order:

        Returns:

        """

        if order == 'desc':
            sorted_list = sorted(item_list, key=lambda x: (x[1], x[0]), reverse=True)
        else:
            sorted_list = sorted(item_list, key=lambda x: (x[1], x[0]), reverse=False)

        return sorted_list

    def segment_search(self, input_segment, keyphrase_list):
        """
        Search for keyphrases in the top-5 PIM segments and return them as final result
        Args:
            input_segment:
            keyphrase_list:

        Returns:

        """
        keywords_list = []
        for tup in keyphrase_list:
            kw = tup[0]
            score = tup[1]

            result = input_segment.find(kw)
            if result > 0:
                keywords_list.append((kw, score))

        sort_list = self.sort_by_value(keywords_list)
        return sort_list

    # TODO
    def export_graph_to_csv(self):
        pass

    def export_graph_to_json(self):
        pass


class TextPreprocess(object):
    def __init__(self):
        pass

    def preprocess_text(self, text, stop_words=False, remove_punct=False, word_tokenize=True, pos=True,
                        filter_by_pos=False, pos_filter=None):
        """
        Preprocess sentences or paragraphs.
        Pipeline followed: text -> tokenize -> pos tagging -> return
        Args:
            pos_filter (list): List of POS tags to filter on
            filter_by_pos (bool): Choose whether to filter by POS tags. Defaults to `False`
            text:
            stop_words (bool):
            remove_punct (bool):
            word_tokenize (bool):
            pos:

        Returns:
            original_tokens (list): list of tokenized words in a sentence.
            word_pos_list (list[tuples]): list of list[(word_token, POS)]
            filtered_word_pos (list[tuples]):
        """
        original_tokens, word_pos_list = text_process.preprocess(text,
                                                                 stop_words=stop_words,
                                                                 remove_punct=remove_punct,
                                                                 word_tokenize=word_tokenize,
                                                                 pos=pos)

        if filter_by_pos:
            filtered_word_pos = self.filter_by_pos(word_pos_tuple=word_pos_list, pos_filter=pos_filter)
        else:
            filtered_word_pos = []

        return original_tokens, word_pos_list, filtered_word_pos

    def filter_by_pos(self, word_pos_tuple, pos_filter=None):
        """
        Filter the (word_token, POS) tuple based on custom POS tags.
        Args:
            word_pos_tuple (list[tuples]): list of list[(word_token, POS)]
            pos_filter (list): list of POS tags to filter on

        Returns:
            filtered_word_pos (list[tuples]): Filtered list of list[(word_token, POS)]
        """

        if pos_filter is None:
            pos_filter = ['ADJ', 'VERB', 'NOUN', 'PROPN', 'FW']

        filtered_word_pos = text_process.get_filtered_pos(sentence=word_pos_tuple, filter_pos=pos_filter)

        return filtered_word_pos
