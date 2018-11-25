package misc;

import dag.Amr;
import dag.Edge;
import dag.Vertex;
import edu.stanford.nlp.ling.Datum;
import gen.GoldTransitions;
import opennlp.tools.ml.model.Event;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

/**
 * A class that provides some useful static methods.
 */
public class StaticHelper {
    // this string is written to each line of the POS tag file generated by
    // posToFile() for which no POS tags were present.
    public static final String POS_TAGGING_ERROR = "POS_TAGGING_ERROR";

    /**
     * Writes a map to a file.
     * @param map the map
     * @param filename the name of the file
     * @throws IOException
     */
    public static <T, S> void mapToFile(Map<T, S> map, String filename)
        throws IOException {
        Properties properties = new Properties();
        properties.putAll(map);
        properties.store(new FileOutputStream(filename), null);
    }

    /**
     * Reads a map from a file.
     * @param filename the name of the file
     * @return the map
     * @throws IOException
     */
    public static <T, S> Map<T, S> mapFromFile(String filename)
        throws IOException {
        Properties properties = new Properties();
        properties.load(new FileInputStream(filename));
        return new HashMap<>((Map<? extends T, ? extends S>) properties);
    }

    /**
     * Writes a list of strings to a file.
     * @param list the list
     * @param filename the name of the file
     * @throws IOException
     */
    public static void listToFile(List<String> list, String filename)
        throws IOException {
        Files.write(Paths.get(filename), list);
    }

    /**
     * Reads a list of strings from a file.
     * @param filename the name of the file
     * @return the list
     * @throws IOException
     */
    public static List<String> listFromFile(String filename)
        throws IOException {
        return Files.readAllLines(Paths.get(filename));
    }

    /**
     * Adds a new line to a file.
     * @param filename the name of the file
     * @param line the content of the line to add
     */
    public static void addLineToFile(String filename, String line) {
        Path path = Paths.get(filename);
        String modLine = System.lineSeparator() + line;

        try (BufferedWriter writer =
                 Files.newBufferedWriter(path, StandardOpenOption.APPEND)) {
            writer.write(modLine);
        } catch (IOException e) {
            System.err.format("IOException: %s%takeBestN", e);
        }
    }

    /**
     * Calculates the median of a collection of integers.
     * @param values the collection of integers
     * @return the median
     */
    public static double median(Collection<Integer> values) {
        List<Integer> sorted = new ArrayList<>(values);
        Collections.sort(sorted);

        if (sorted.size() % 2 == 0) {
            return (sorted.get(sorted.size() / 2)
                       + sorted.get(sorted.size() / 2 - 1))
                / 2d;
        }

        return sorted.get(sorted.size() / 2);
    }

    /**
     * Generates all possible permutations for a list of edges.
     * <b>Example:</b> If {@code list = (e1, e2, e3)}, then this function
     * returns {@code ((e1, e2, e3), (e1, e3, e2), (e2, e1, e3), (e2, e3, e1),
     * (e3, e1, e2), (e3, e2, e1))}.
     * @param list the list of edges
     * @return the list of permutations
     */
    public static List<List<Edge>> listPermutations(List<Edge> list) {
        if (list.size() == 0) {
            List<List<Edge>> result = new ArrayList<>();
            result.add(new ArrayList<>());
            return result;
        }

        List<List<Edge>> ret = new ArrayList<>();

        Edge firstElement = list.remove(0);

        List<List<Edge>> recursiveReturn = listPermutations(list);
        for (List<Edge> li : recursiveReturn) {
            for (int index = 0; index <= li.size(); index++) {
                List<Edge> temp = new ArrayList<>(li);
                temp.add(index, firstElement);
                ret.add(temp);
            }
        }
        return ret;
    }

    /**
     * Sorts a map with comparable values by its values.
     * @param map the map
     * @return the sorted map
     */
    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(
        Map<K, V> map) {
        return map.entrySet()
            .stream()
            .sorted(Entry.comparingByValue(Collections.reverseOrder()))
            .collect(Collectors.toMap(Entry::getKey, Entry::getValue,
                (e1, e2) -> e1, LinkedHashMap::new));
    }

    /**
     * Extracts from a corpus of AMR graphs a map mapping each concept to all
     * observed realizations (TAB separated).
     * @param amrs the list of AMR graphs from which the map should be extracted
     * @return the (concept → realization1 TAB realization_2 TAB ... TAB
     * realization_n) map
     */
    public static Map<String, String> getConceptRealizationMap(List<Amr> amrs) {
        Map<String, String> ret = new HashMap<>();
        Map<String, Set<String>> crMap = new HashMap<>();

        for (Amr amr : amrs) {
            for (Vertex vertex : amr.dag) {
                String realization = GoldTransitions.getGoldRealization(
                    amr, vertex.getInstanceEdge());

                if (!crMap.containsKey(vertex.getInstance())) {
                    crMap.put(vertex.getInstance(), new HashSet<>());
                }

                crMap.get(vertex.getInstance()).add(realization);
            }
        }

        for (String key : crMap.keySet()) {
            ret.put(key, String.join("\t", crMap.get(key)));
        }

        return ret;
    }

    /**
     * Extracts from a corpus of AMR graphs a map mapping each non-PropBank
     * concept to the POS tag observed most often.
     * @param amrs the list of AMR graphs from which the map should be extracted
     * @return the (concept → pos) map
     */
    public static Map<String, String> getBestPosTagsMap(List<Amr> amrs) {
        Map<String, String> ret = new HashMap<>();
        Map<String, Map<String, Integer>> cpMap = new HashMap<>();

        // get a map that counts for each (concept, pos) pair the number of
        // occurrences
        for (Amr amr : amrs) {
            for (Vertex vertex : amr.dag) {
                if (vertex.isPropbankEntry())
                    continue;

                String key = vertex.getInstance();

                if (!cpMap.containsKey(key)) {
                    cpMap.put(key, new HashMap<>());
                }

                Map<String, Integer> countMap = cpMap.get(key);
                countMap.put(vertex.getPos(),
                    countMap.getOrDefault(vertex.getPos(), 0) + 1);
            }
        }

        // for each key of cpMap, get the key with the maximum value in
        // cpMap.get(key)
        for (String key : cpMap.keySet()) {
            ret.put(key,
                Collections
                    .max(cpMap.get(key).entrySet(), Entry.comparingByValue())
                    .getKey());
        }

        return ret;
    }

    /**
     * Extracts from a corpus of AMR graphs a map mapping each (concept, pos)
     * pair (TAB separated) to the realization observed most often.
     * @param amrs the list of AMR graphs from which the map should be extracted
     * @return the ((concept, pos) → realization) map
     */
    public static Map<String, String> getConceptPosBestRealizationMap(
        List<Amr> amrs) {
        Map<String, String> ret = new HashMap<>();
        Map<String, Map<String, Integer>> cpMap = new HashMap<>();

        // get a map that counts for each (concept, pos, realization) pair the
        // number of occurrences
        for (Amr amr : amrs) {
            for (Vertex vertex : amr.dag) {
                String realization = GoldTransitions.getGoldRealization(
                    amr, vertex.getInstanceEdge());
                String key = vertex.getInstance() + "\t" + vertex.getPos();

                if (!cpMap.containsKey(key)) {
                    cpMap.put(key, new HashMap<>());
                }

                Map<String, Integer> countMap = cpMap.get(key);
                countMap.put(
                    realization, countMap.getOrDefault(realization, 0) + 1);
            }
        }

        // for each key of cpMap, get the key with the maximum value in
        // cpMap.get(key)
        for (String key : cpMap.keySet()) {
            ret.put(key,
                Collections
                    .max(cpMap.get(key).entrySet(), Entry.comparingByValue())
                    .getKey());
        }

        return ret;
    }

    /**
     * Extracts from a corpus of AMR graphs a map mapping each mergeable
     * (parent, child) pair (TAB separated) to the corresponding merged vertex
     * observed most often.
     * @param amrs the list of AMR graphs from which the map should be extracted
     * @return the ((parent, child) → merge) map
     */
    public static Map<String, String> getMergeMap(List<Amr> amrs) {
        Map<String, String> ret = new HashMap<>();
        Map<String, Map<String, Integer>> mergeCounts = new HashMap<>();

        for (Amr amr : amrs) {
            for (Vertex v : amr.dag) {
                if (!v.getIncomingEdges().isEmpty()) {
                    String instance = v.getInstance();
                    String parentInstance =
                        v.getIncomingEdges().get(0).getFrom().getInstance();
                    String result = GoldTransitions.getGoldMerge(amr, v);

                    if (result != null) {
                        String key = parentInstance + "\t" + instance;

                        if (!mergeCounts.containsKey(key)) {
                            mergeCounts.put(key, new HashMap<>());
                        }

                        Map<String, Integer> resultCount = mergeCounts.get(key);
                        resultCount.put(
                            result, resultCount.getOrDefault(result, 0) + 1);
                    }
                }
            }
        }

        for (String key : mergeCounts.keySet()) {
            String result = Collections
                                .max(mergeCounts.get(key).entrySet(),
                                    Entry.comparingByValue())
                                .getKey();
            ret.put(key, result);
        }

        return ret;
    }

    /**
     * Extracts from a corpus of AMR graphs a map mapping each mergeable
     * (sibling, sibling) pair (TAB separated) to the corresponding merged
     * vertex observed most often.
     * This provides almost the same functionality as {@link #getMergeMap(List)}
     * except that we do not know what two siblings are to be merged. As a
     * consequence this information has to be passed onto this function as well
     * (besides the resulting merge).
     * @param amrs the list of AMR graphs from which the map should be extracted
     * @return the ((sibling, sibling) → merge) map
     */
    public static Map<String, String> getMergeSiblingMap(List<Amr> amrs) {
        Map<String, String> ret = new HashMap<>();
        Map<String, Map<String, Integer>> mergeCounts = new HashMap<>();

        for (Amr amr : amrs) {
            for (Vertex v : amr.dag) {
                // only for vertices that have a parent
                if (!v.getIncomingEdges().isEmpty()) {
                    // determine if there are 2 siblings that can be merged and
                    // return these vertices' instances alongside their merge
                    List<String> mergePair =
                        GoldTransitions.getGoldMergeSibling(amr, v);

                    if (mergePair != null) {
                        // the 2 siblings that are to be merged
                        String key = mergePair.get(0);
                        // their merge
                        String result = mergePair.get(1);

                        if (!mergeCounts.containsKey(key)) {
                            mergeCounts.put(key, new HashMap<>());
                        }

                        Map<String, Integer> resultCount = mergeCounts.get(key);
                        resultCount.put(
                            result, resultCount.getOrDefault(result, 0) + 1);
                    }
                }
            }
        }

        for (String key : mergeCounts.keySet()) {
            String result = Collections
                                .max(mergeCounts.get(key).entrySet(),
                                    Entry.comparingByValue())
                                .getKey();
            ret.put(key, result);
        }

        return ret;
    }

    /**
     * Extracts from a corpus of AMR graphs a duplicate-free list of all
     * observed concepts.
     * @param amrs the list of AMR graphs from which the list should be
     * extracted
     * @return the list of concepts
     */
    public static List<String> getConceptList(List<Amr> amrs) {
        Set<String> ret = new HashSet<>();
        for (Amr amr : amrs) {
            for (Vertex v : amr.dag) {
                if (v.isTranslatable()) {
                    ret.add(v.getInstance());
                }
            }
        }
        return new ArrayList<>(ret);
    }

    /**
     * Checks whether the concept represented by a vertex is numeric and not
     * equal to one.
     * @param v the vertex to check
     * @return {@code :NUMERIC} if the vertex represents a number not equal to
     * one, {@code v.getInstance()} otherwise
     */
    public static String getInstanceOrNumeric(Vertex v) {
        String inst = v.getInstance();
        if (inst.matches("[0-9.]+") && !inst.equals("1") && !inst.equals("1.0"))
            return ":NUMERIC";
        return inst;
    }

    /**
     * Extracts from a corpus of AMR graphs a map mapping each tab-separated
     * triple of <ol> <li>an AMR concept,</li> <li>a name for this concept
     * and</li> <li>the information whether the concept is deleted, left or
     * right of the name in the reference realization</li>
     * </ol>
     * to the number of times this triple has been observed.
     * @param amrs the list of AMR graphs from which the map should be extracted
     * @return the ((instance, name, position) → count) map
     */
    public static Map<String, String> getNamedEntityMap(List<Amr> amrs) {
        Map<String, Integer> namedEntityMap = new HashMap<>();

        for (Amr amr : amrs) {
            for (Vertex v : amr.dag) {
                if (!v.name.isEmpty()) {
                    if (amr.alignment.containsKey(v.getInstanceEdge())) {
                        List<Integer> align = new ArrayList<>(
                            amr.alignment.get(v.getInstanceEdge()));
                        Collections.sort(align);

                        int maxIndex = 0;

                        for (int i = 1; i < align.size(); i++) {
                            if (align.get(i - 1) + 1 == align.get(i)) {
                                maxIndex = i;
                            }
                        }

                        align = align.subList(0, maxIndex + 1);

                        String k1 =
                            v.name.toLowerCase() + "\t" + v.getInstance();
                        String k2 = v.getInstance();

                        for (String key : Arrays.asList(k1, k2)) {
                            String realization =
                                String
                                    .join(" ",
                                        align.stream()
                                            .map(j -> amr.sentence[j])
                                            .collect(Collectors.toList()))
                                    .toLowerCase();

                            String order1 = v.name.toLowerCase() + " "
                                + v.getInstance().toLowerCase();
                            String order2 = v.getInstance().toLowerCase() + " "
                                + v.name.toLowerCase();

                            if (realization.equals(order1)) {
                                String composedKey =
                                    key + "\t" + InstPosition.RIGHT.getValue();
                                namedEntityMap.put(composedKey,
                                    namedEntityMap.getOrDefault(composedKey, 0)
                                        + 1);
                            } else if (realization.equals(order2)) {
                                String composedKey =
                                    key + "\t" + InstPosition.LEFT.getValue();
                                namedEntityMap.put(composedKey,
                                    namedEntityMap.getOrDefault(composedKey, 0)
                                        + 1);
                            } else if (realization.equals(
                                           v.name.toLowerCase())) {
                                String composedKey =
                                    key + "\t" + InstPosition.DELETE.getValue();
                                namedEntityMap.put(composedKey,
                                    namedEntityMap.getOrDefault(composedKey, 0)
                                        + 1);
                            }
                        }
                    }
                }
            }
        }

        Map<String, String> ret = new HashMap<>();
        for (String key : namedEntityMap.keySet()) {
            ret.put(key, namedEntityMap.get(key) + "");
        }

        return ret;
    }

    /**
     * Converts a datum as used by the Stanford MaxEntModel to an event as used
     * by the OpenNLP MaxEnt MaxEntModel
     * @param datum the datum in Stanford MaxEntModel format
     * @return the event in OpenNLP MaxEnt MaxEntModel format
     */
    public static Event toEvent(Datum<String, String> datum) {
        Collection<String> features = datum.asFeatures();
        return new Event(
            datum.label(), features.toArray(new String[features.size()]));
    }

    /**
     * see {@link StaticHelper#toEvent(Datum)}
     */
    public static List<Event> toEvents(List<Datum<String, String>> data) {
        return data.stream()
            .map(StaticHelper::toEvent)
            .collect(Collectors.toList());
    }

    /**
     * Writes the POS tags for a list of AMR graphs to a file.
     * @param amrs the list of AMR graphs; each AMR graph must already contain
     * the corresponding POS tags
     * @param path the name of the file
     */
    public static void posToFile(List<Amr> amrs, String path)
        throws IOException {
        List<String> tags = new ArrayList<>();
        for (Amr amr : amrs) {
            try {
                tags.add(String.join("\t", amr.pos));
            } catch (NullPointerException e) {
                Debugger.printlnErr("error at sentence = " + amr.sentence
                    + ", pos = " + amr.pos);
                tags.add(POS_TAGGING_ERROR);
            }
        }
        Files.write(Paths.get(path), tags);
    }

    /**
     * Checks wheter a set of integers is contiguous, i.e. there are some
     * natural numbers takeBestN, m such that the set can be written as
     * {takeBestN, takeBestN+1, takeBestN+2, ..., takeBestN+m-1, takeBestN+m}.
     * @param s the set of integers
     * @return whether the set is contiguous
     */
    public static boolean isContiguous(Set<Integer> s) {
        int min = s.stream().mapToInt(i -> i).min().getAsInt();
        int max = s.stream().mapToInt(i -> i).max().getAsInt();
        return s.size() == max - min + 1;
    }

    /**
     * Returns the leftmost contiguous subset of a set of integers. For example,
     * if s = {2,3,4,6,8,9,10,11}, this function returns the sequence [2,3,4].
     * @param s the set of integers
     * @return the leftmost contiguous subset of s
     */
    public static List<Integer> getLeftmostContiguous(Set<Integer> s) {
        List<Integer> l = new ArrayList<>(s);
        Collections.sort(l);

        int maxIndex = 0;

        for (int i = 1; i < l.size(); i++) {
            if (l.get(i - 1) + 1 == l.get(i)) {
                maxIndex = i;
            }
        }

        return l.subList(0, maxIndex + 1);
    }

    /**
     * Removes all words occuring twice or more times in a row in a
     * space-separated sequence of words
     * @param sent the sequence of words
     * @return the same sequence where duplicate words are removed
     */
    public static String removeDuplicateWords(String sent) {
        List<String> sentence = new ArrayList<>(Arrays.asList(sent.split(" ")));

        // check if the same word appears twice
        for (int i = 0; i < sentence.size() - 1; i++) {
            if (sentence.get(i).equals(sentence.get(i + 1))) {
                sentence.set(i, "");
            }
        }

        sentence.removeIf(String::isEmpty);
        return String.join(" ", sentence);
    }

    public enum InstPosition {
        LEFT("l"),
        RIGHT("r"),
        DELETE("d");
        String value;
        InstPosition(String value) {
            this.value = value;
        }
        public String getValue() {
            return value;
        }
    }
}
