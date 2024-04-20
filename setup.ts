import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { createRetrieverTool } from "langchain/tools/retriever";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { pull } from "langchain/hub";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

const directoryLoader = new DirectoryLoader(
    "ap_testbooks",
    {
      ".pdf": (path: string) => new PDFLoader(path),
    }
  );

const searchTool = new TavilySearchResults({apiKey: "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"});

const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);

const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
[
    "system",
    "Answer the user's questions based on the below context:\n\n{context}",
],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
]);

const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}`);



const outputParser = new StringOutputParser();

export async function setupLLM() {

    const chatModel = new ChatOpenAI({
        apiKey: "sk-proj-4CCSBMgSGCMuPctnvLikT3BlbkFJnvb2RgMcL0NEa8EM6pex",
    });
    
    const docs = await directoryLoader.load();
    console.log(docs.length);
    console.log(docs[0].pageContent.length);
    const embeddings = new OpenAIEmbeddings({apiKey: "sk-proj-4CCSBMgSGCMuPctnvLikT3BlbkFJnvb2RgMcL0NEa8EM6pex"});
    const splitter = new RecursiveCharacterTextSplitter({chunkSize: 1000, chunkOverlap: 200});

    const splitDocs = await splitter.splitDocuments(docs);
    const vectorstore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
      );
    console.log(splitDocs.length);
    console.log(splitDocs[0].pageContent.length);

    const documentChain = await createStuffDocumentsChain({
        llm: chatModel,
        prompt,
      });
    
    const retriever = vectorstore.asRetriever();

    const retrievalChain = await createRetrievalChain({
        combineDocsChain: documentChain,
        retriever,
    });
    const historyAwareRetrieverChain = await createHistoryAwareRetriever({
        llm: chatModel,
        retriever,
        rephrasePrompt: historyAwarePrompt,
      });
    
    const historyAwareCombineDocsChain = await createStuffDocumentsChain({
        llm: chatModel,
        prompt: historyAwareRetrievalPrompt,
    });
    const conversationalRetrievalChain = await createRetrievalChain({
        retriever: historyAwareRetrieverChain,
        combineDocsChain: historyAwareCombineDocsChain,
    });

    const retrieverTool = await createRetrieverTool(retriever, {
        name: "ap_search",
        description:
          "Search for information about specific topics in any AP exam. For any questions about AP exam structure and topics, you must use this tool!",
      });

    const tools = [retrieverTool, searchTool];

    const agentPrompt = await pull<ChatPromptTemplate>(
        "hwchase17/openai-functions-agent"
    );
    
    const agentModel = new ChatOpenAI({
        model: "gpt-3.5-turbo-1106",
        temperature: 0,
        apiKey: "sk-proj-4CCSBMgSGCMuPctnvLikT3BlbkFJnvb2RgMcL0NEa8EM6pex",
    });
    
    const agent = await createOpenAIFunctionsAgent({
        llm: agentModel,
        tools,
        prompt: agentPrompt,
    });
    
    return new AgentExecutor({agent, tools, verbose: false});
}

class Singleton {
    private static instance: AgentExecutor | null = null;

    static async getInstance(): Promise<AgentExecutor> {
        if (!Singleton.instance) {
            Singleton.instance = await setupLLM();
        }
        return Singleton.instance;
    }
}

export const agent = Singleton.getInstance();

export async function main(userInput: string, agent: AgentExecutor) {
    
    const agentResult = await agent.invoke({
        input: userInput,
    });
    console.log(agentResult.output);
}