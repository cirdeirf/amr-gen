package ml;

import dag.*;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import gen.GoldSyntacticAnnotations;
import gen.GoldTransitions;
import misc.StaticHelper;
import misc.WordLists;

import java.util.*;
import java.util.stream.Collectors;

/**
 * This class implements a maximum entropy model for articles. See {@link
 * StanfordMaxentModelImplementation} for further details on the implemented
 * methods. The features used by {@link DenomMaxentModel#toDatumList(Amr,
 * Vertex, boolean)} are explained in the thesis.
 */
public class DenomMaxentModel extends StanfordMaxentModelImplementation {
    @Override
    public List<Datum<String, String>> toDatumList(
        Amr amr, Vertex vertex, boolean forTesting) {
        return toDatumList(amr, vertex, forTesting, null, null);
    }

    public List<Datum<String, String>> toDatumList(Amr amr, Vertex vertex,
        boolean forTesting, String numberPrediction, String realization) {
        boolean noArticlePossible =
            !vertex.getPos().equals("NN") && vertex.name.isEmpty();
        if (noArticlePossible || vertex.isDeleted() || vertex.isLink())
            return Collections.emptyList();

        String result =
            GoldSyntacticAnnotations.getGoldDenominator(amr, vertex);

        Edge instanceEdge = vertex.getInstanceEdge();

        List<Edge> outEdges = new ArrayList<>(vertex.getOutgoingEdges());
        outEdges.remove(instanceEdge);

        List<String> outWithPos = new ArrayList<>();

        for (Edge e : outEdges) {
            if (!e.getTo().isPropbankEntry()) {
                outWithPos.add(e.getLabel() + "," + e.getTo().getPos());
            }
        }

        List<String> outStrings = outEdges.stream()
                                      .map(e -> e.getLabel())
                                      .distinct()
                                      .collect(Collectors.toList());

        ListFeature argFeatures = new ListFeature("argFeatures");
        ListFeature argLinkFeatures = new ListFeature("argLinkFeatures");

        for (int i = 0; i < 4; i++) {
            if (outStrings.contains(":ARG" + i)) {
                argFeatures.add(i + "pr");
            } else {
                argFeatures.add(i + "no");
            }
        }

        for (int i = 0; i < 4; i++) {
            int finalI = i;
            Optional<Edge> outE =
                outEdges.stream()
                    .filter(e -> e.getLabel().equals(":ARG" + finalI))
                    .findFirst();
            if (outE.isPresent()) {
                if (outE.get().getTo().isLink()) {
                    argLinkFeatures.add(i + "link");
                } else {
                    argLinkFeatures.add(i + "pr");
                }
            } else {
                argLinkFeatures.add(i + "no");
            }
        }

        String instance = vertex.getInstance();
        String inLabel, parentInstance;

        ListFeature parentInLabels = new ListFeature("parentInLabels");
        ListFeature parentPosTags = new ListFeature("parentPosTags");
        ListFeature parentPropEntries = new ListFeature("parentPropEntries");

        Vertex currentVertex = vertex;
        Vertex parentVertex = null, grandparentVertex = null;

        int distanceToRoot = 0;
        while (!currentVertex.getIncomingEdges().isEmpty()) {
            if (distanceToRoot > 0) {
                parentInLabels.add(currentVertex.getIncomingEdges().isEmpty()
                        ? ":ROOT"
                        : currentVertex.getIncomingEdges().get(0).getLabel());
                parentPropEntries.add(
                    currentVertex.isPropbankEntry() + "_d:" + distanceToRoot);
                parentPosTags.add(currentVertex.isPropbankEntry()
                        ? ":PROP"
                        : currentVertex.getPos());
            }

            if (currentVertex == vertex) {
                parentVertex =
                    currentVertex.getIncomingEdges().get(0).getFrom();
            } else if (currentVertex == parentVertex) {
                grandparentVertex =
                    currentVertex.getIncomingEdges().get(0).getFrom();
            }

            distanceToRoot++;
            currentVertex = currentVertex.getIncomingEdges().get(0).getFrom();
        }

        String parentPos;

        inLabel = ":ROOT";
        parentInstance = ":ROOT";
        parentPos = ":ROOT";

        String grandparentInstance = ":ROOT";
        String parentInLabel = ":ROOT";
        String grandparentPos = ":ROOT";

        if (parentVertex != null) {
            parentInstance = StaticHelper.getInstanceOrNumeric(parentVertex);
            parentPos = parentVertex.isPropbankEntry() ? ":PROP"
                                                       : parentVertex.getPos();
            inLabel = vertex.getIncomingEdges().get(0).getLabel();
        }

        if (grandparentVertex != null) {
            grandparentInstance = grandparentVertex.getInstance();
            grandparentPos = grandparentVertex.isPropbankEntry()
                ? ":PROP"
                : grandparentVertex.getPos();
            parentInLabel = parentVertex.getIncomingEdges().get(0).getLabel();
        }

        List<String> argOfFeatures = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            if (inLabel.equals(":ARG" + i + "-of")) {
                argOfFeatures.add(i + "pr");
            } else {
                argOfFeatures.add(i + "npr");
            }
        }

        String lemma = vertex.isPropbankEntry()
            ? instance.substring(0, instance.lastIndexOf('-'))
            : instance;

        List<String> neighbourLabels = new ArrayList<>();
        List<String> neighbourInstances = new ArrayList<>();
        List<String> neighbourPosTags = new ArrayList<>();
        List<String> neighbourLabelPosTags = new ArrayList<>();
        if (parentVertex != null) {
            for (Edge e : parentVertex.getOutgoingEdges()) {
                if (!e.isInstanceEdge()
                    && e != vertex.getIncomingEdges().get(0)) {
                    neighbourLabels.add(e.getLabel());
                    neighbourInstances.add(
                        StaticHelper.getInstanceOrNumeric(e.getTo()));
                    neighbourPosTags.add(e.getTo().isPropbankEntry()
                            ? ":PROP"
                            : e.getTo().getPos());
                    neighbourLabelPosTags.add(e.getLabel() + ","
                        + (e.getTo().isPropbankEntry() ? ":PROP"
                                                       : e.getTo().getPos()));
                }
            }
        }

        List<String> outLabelPosTag = new ArrayList<>();

        for (Edge e : outEdges) {
            outLabelPosTag.add(e.getLabel() + "-"
                + (e.getTo().isPropbankEntry() ? ":PROP" : e.getTo().getPos()));
        }

        List<String> allParentConcepts = new ArrayList<>();
        List<String> allInLabels = new ArrayList<>();
        List<String> allPosInLabels = new ArrayList<>();
        allParentConcepts.add(parentInstance);
        allInLabels.add(inLabel);
        allPosInLabels.add(inLabel + "," + parentPos);

        if (parentVertex != null) {
            for (Vertex v : amr.dag) {
                if (v.isLink() && v.annotation.original == vertex) {
                    if (!v.getIncomingEdges().isEmpty()) {
                        Vertex newParent =
                            v.getIncomingEdges().get(0).getFrom();
                        allParentConcepts.add(newParent.getInstance());
                        allInLabels.add(v.getIncomingEdges().get(0).getLabel());
                        allPosInLabels.add(
                            v.getIncomingEdges().get(0).getLabel() + ","
                            + newParent.getInstance());
                    }
                }
            }
        }

        boolean hasInverseLabel = inLabel.endsWith("-of");
        boolean hasArgLabel = inLabel.startsWith(":ARG");

        String number;
        String real;

        if (!forTesting) {
            number = GoldSyntacticAnnotations.getGoldNumber(amr, vertex);
            real = GoldTransitions.getGoldRealization(
                amr, vertex.getInstanceEdge());
        } else {
            if (numberPrediction != null) {
                number = numberPrediction;
            } else {
                number = vertex.predictions.get("number").get(0).getValue();
            }
            real = realization;
        }

        Optional<Edge> modEdge = outEdges.stream()
                                     .filter(e -> e.getLabel().equals(":mod"))
                                     .findAny();
        String modPos = "no_mod";
        String modInst = "no_mod";
        if (modEdge.isPresent()) {
            Edge me = modEdge.get();
            modPos = me.getTo().isPropbankEntry() ? "propEntry"
                                                  : me.getTo().getPos();
            modInst = me.getTo().getInstance();
        }

        Set<String> outPosTag = new HashSet<>();

        for (Edge e : outEdges) {
            outLabelPosTag.add(e.getLabel() + "-"
                + (e.getTo().isPropbankEntry() ? ":PROP" : e.getTo().getPos()));
            outPosTag.add(
                (e.getTo().isPropbankEntry() ? ":PROP" : e.getTo().getPos()));
        }

        List<IndicatorFeature> features = new ArrayList<>();

        features.add(new StringFeature("realization", real));
        features.add(new StringFeature("number", number));
        features.add(new StringFeature(
            "polarityPresent", outStrings.contains(":polarity")));
        features.add(new StringFeature("modPOS", modPos));
        features.add(new StringFeature("modInst", modInst));
        features.add(new StringFeature("number-instance", number + instance));
        features.add(new StringFeature("number-inLabel", number + inLabel));
        features.add(new StringFeature(
            "inlabel-outEmpty", inLabel + outEdges.isEmpty()));
        features.add(new StringFeature("parentInst", parentInstance));
        features.add(new StringFeature("name", vertex.name));
        features.add(new StringFeature(
            "nameOrInstance", vertex.name.isEmpty() ? instance : vertex.name));
        features.add(new StringFeature("instance", instance));
        features.add(new StringFeature("inLabel", inLabel));
        features.add(new StringFeature("name-inLabel", vertex.name + inLabel));
        features.add(new StringFeature("number-modPOS", number + modPos));
        features.add(
            new StringFeature("modInst-nameOrInstance", modInst + instance));
        features.add(new StringFeature("modInst-instance", modInst + instance));
        features.add(new StringFeature(
            "hasName-instance", vertex.name.isEmpty() + instance));
        features.add(new StringFeature("isCountry",
            WordLists.countryforms.keySet().contains(vertex.name)));
        features.add(new ListFeature("outPosTag", new ArrayList<>(outPosTag)));
        features.add(new ListFeature("outLabel-posTag", outLabelPosTag));
        features.add(new ListFeature("childrenWithLabels",
            outEdges.stream()
                .map(e -> e.getLabel() + e.getTo().getInstance())
                .collect(Collectors.toList())));
        features.add(new StringFeature("instance", instance));
        features.add(new StringFeature("lemma",
            lemma + (parentVertex == null ? ":ROOT" : parentVertex.mode)));
        features.add(new StringFeature(
            "instance-outEmpty", instance + outEdges.isEmpty()));
        features.add(new StringFeature("parentInst", parentInstance));
        features.add(
            new StringFeature("parentInst-inLabel", parentInstance + inLabel));
        features.add(new StringFeature("inLabel", inLabel));
        features.add(new StringFeature("outSize", outEdges.size()));
        features.add(new StringFeature("depth", vertex.subtreeSize()));
        features.add(new StringFeature("numberOfArgs",
            outEdges.stream()
                    .filter(e -> e.getLabel().matches(":ARG[0-9]"))
                    .count()
                + ""));
        features.add(
            new ListFeature("outLabelPosTag", outLabelPosTag)
                .composeWith(new StringFeature("inLabel", inLabel), "*c2"));
        features.add(new StringFeature("parentInst-grandparentInst",
            parentInstance + grandparentInstance));
        features.add(
            new ListFeature("outLabels", outStrings)
                .composeWith(new StringFeature("inLabel", inLabel), "*c1"));
        features.add(new ListFeature("outLabels", outStrings));
        features.add(new StringFeature("instance", instance));
        features.add(new StringFeature("lemma", lemma));
        features.add(new ListFeature("argOfFeatures", argOfFeatures));
        features.add(new ListFeature("neighbourLabels", neighbourLabels));
        features.add(new ListFeature("neighbourInstances", neighbourInstances));
        features.add(new ListFeature("neighbourPosTags", neighbourPosTags));
        features.add(new ListFeature("neighbourLabels", neighbourLabels));
        features.add(
            new ListFeature("neighbourLabelPosTags", neighbourLabelPosTags));
        features.add(new ListFeature("children",
            outEdges.stream()
                .filter(e -> !e.isInstanceEdge())
                .map(e -> StaticHelper.getInstanceOrNumeric(e.getTo()))
                .collect(Collectors.toList())));
        features.add(new ListFeature("nonLinkChildren",
            outEdges.stream()
                .filter(e -> !e.isInstanceEdge() && !e.getTo().isLink())
                .map(e -> StaticHelper.getInstanceOrNumeric(e.getTo()))
                .collect(Collectors.toList())));
        features.add(
            new StringFeature("neighbourSize", neighbourLabels.size()));
        features.add(
            new StringFeature("noNeighbours", neighbourLabels.isEmpty()));
        features.add(new StringFeature("parentPos", parentPos));
        features.add(new ListFeature("allParentInsts", allParentConcepts));
        features.add(new ListFeature("allInLabels", allInLabels));
        features.add(new ListFeature("allPosInLabels", allPosInLabels));
        features.add(new StringFeature("grandparentPos", grandparentPos));
        features.add(new StringFeature("grandparentInst", grandparentInstance));
        features.add(new StringFeature("parentInLabel", parentInLabel));
        features.add(parentPosTags);
        features.add(parentInLabels);
        features.add(parentPropEntries);
        features.add(new StringFeature("distToRoot", distanceToRoot));
        features.add(new StringFeature("inLabel", inLabel));
        features.add(new StringFeature("outSize", outEdges.size()));
        features.add(new StringFeature("outEmpty", outEdges.isEmpty()));
        features.add(new ListFeature("outLabelPosTag", outLabelPosTag));
        features.add(new ListFeature("childrenWithLabels",
            outEdges.stream()
                .map(e
                    -> e.getLabel()
                        + StaticHelper.getInstanceOrNumeric(e.getTo()))
                .collect(Collectors.toList())));
        features.add(argFeatures);
        features.add(argLinkFeatures);
        features.add(new StringFeature(
            "parentMode", parentVertex == null ? ":ROOT" : parentVertex.mode));
        features.add(new StringFeature("hasInverseLabel", hasInverseLabel));
        features.add(new StringFeature(
            "hasInvArgFeature", (hasInverseLabel && hasArgLabel)));
        featureManager.addAllUnaries(features);

        List<String> context = featureManager.toContext();
        return Collections.singletonList(new BasicDatum<>(context, result));
    }

    @Override
    public void applyModification(
        Amr amr, Vertex vertex, List<Prediction> predictions) {
        vertex.predictions.put("article", predictions);
    }
}
