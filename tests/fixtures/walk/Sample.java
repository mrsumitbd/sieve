package demo;

import java.util.List;
import java.util.function.Supplier;

/** A documented class. */
@Deprecated
public class Sample extends Base implements Runnable, Comparable<Sample> {

    private final Supplier<Integer> field = new Supplier<Integer>() {
        public Integer get() { return 1; }
    };

    static class StaticNested {
        void inNested() {}

        class Inner {
            int deep() { return 2; }
        }
    }

    public Sample() {}

    /** Runs. */
    @Override
    public void run() {
        Runnable r = new Runnable() {
            public void run() {}
        };
        class LocalClass {
            void local() {}
        }
        List<String> xs = List.of();
        xs.forEach(x -> System.out.println(x));
    }

    public int compareTo(Sample o) { return 0; }

    interface Shape {
        double area();
        default String label() { return "shape"; }
    }

    enum Color {
        RED { void paint() {} },
        GREEN;
        void common() {}
    }

    record Pair(int a, int b) {
        int sum() { return a + b; }
    }

    @interface Marker {}
}

class Second {
    Second(int x) {}
    static void util(String... args) {}
}
