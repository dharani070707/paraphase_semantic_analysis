import { useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";

type ResultData = {
  similarity: number;
  paraphrase: boolean;
};

export default function Result() {
  const location = useLocation();
  const { text1, text2 } = location.state || {};

  const [data, setData] = useState<ResultData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchResult = async () => {
      try {
        const res = await axios.post("http://localhost:5000/analyze", {
          text1,
          text2,
        });

        setData(res.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, []);

  return (
    <div className="min-h-screen bg-[#030304] text-white flex items-center justify-center p-6">
      {/* Card */}
      <div className="w-full max-w-3xl bg-white/[0.03] border border-white/10 backdrop-blur-xl rounded-3xl p-10 shadow-2xl">
        {/* Header */}
        <h1 className="text-3xl font-bold mb-6 text-center">
          🔍 Analysis Result
        </h1>

        {/* Input Display */}
        <div className="space-y-4 mb-8 text-sm text-gray-400">
          <div>
            <p className="text-gray-500">Sentence 1</p>
            <p className="text-white">{text1}</p>
          </div>

          <div>
            <p className="text-gray-500">Sentence 2</p>
            <p className="text-white">{text2}</p>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div className="flex justify-center items-center py-10">
            <div className="w-10 h-10 border-4 border-white border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {/* Result */}
        {data && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-8 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-2xl text-white flex flex-col items-center gap-6"
          >
            {/* Similarity */}
            <div className="text-center">
              <p className="text-sm opacity-70">Similarity Score</p>
              <h2 className="text-6xl font-bold">
                {(data.similarity * 100).toFixed(0)}%
              </h2>
            </div>

            {/* Divider */}
            <div className="w-full h-[1px] bg-white/20" />

            {/* Paraphrase */}
            <div className="text-center">
              <p className="text-sm opacity-70">Paraphrase</p>
              <h2 className="text-2xl font-semibold">
                {data.paraphrase
                  ? "Yes, both the sentences are semantically similar"
                  : "Not semantically similar both sentences are different"}
              </h2>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
